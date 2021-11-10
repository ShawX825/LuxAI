import sys
import time
from functools import partial  # pip install functools

import numpy as np
from gym import spaces

from luxai2021.env.agent import Agent
from luxai2021.game.actions import *
from luxai2021.game.game_constants import GAME_CONSTANTS
from luxai2021.game.position import Position
from collections import OrderedDict

# Constants
OFFSET_OPPONENT = 10 # Offset added to oppontent types
CITIES = 2 # Special unit type code for cities

# https://codereview.stackexchange.com/questions/28207/finding-the-closest-point-to-a-list-of-points
def closest_node(node, nodes):
    dist_2 = np.sum((nodes - node) ** 2, axis=1)
    return np.argmin(dist_2)
def furthest_node(node, nodes):
    dist_2 = np.sum((nodes - node) ** 2, axis=1)
    return np.argmax(dist_2)

def smart_transfer_to_nearby(game, team, unit_id, unit, target_type_restriction=None, **kwarg):
    """
    Smart-transfers from the specified unit to a nearby neighbor. Prioritizes any
    nearby carts first, then any worker. Transfers the resource type which the unit
    has most of. Picks which cart/worker based on choosing a target that is most-full
    but able to take the most amount of resources.

    Args:
        team ([type]): [description]
        unit_id ([type]): [description]

    Returns:
        Action: Returns a TransferAction object, even if the request is an invalid
                transfer. Use TransferAction.is_valid() to check validity.
    """

    # Calculate how much resources could at-most be transferred
    resource_type = None
    resource_amount = 0
    target_unit = None

    if unit != None:
        for type, amount in unit.cargo.items():
            if amount > resource_amount:
                resource_type = type
                resource_amount = amount

        # Find the best nearby unit to transfer to
        unit_cell = game.map.get_cell_by_pos(unit.pos)
        adjacent_cells = game.map.get_adjacent_cells(unit_cell)

        
        for c in adjacent_cells:
            for id, u in c.units.items():
                # Apply the unit type target restriction
                if target_type_restriction == None or u.type == target_type_restriction:
                    if u.team == team:
                        # This unit belongs to our team, set it as the winning transfer target
                        # if it's the best match.
                        if target_unit is None:
                            target_unit = u
                        else:
                            # Compare this unit to the existing target
                            if target_unit.type == u.type:
                                # Transfer to the target with the least capacity, but can accept
                                # all of our resources
                                if( u.get_cargo_space_left() >= resource_amount and 
                                    target_unit.get_cargo_space_left() >= resource_amount ):
                                    # Both units can accept all our resources. Prioritize one that is most-full.
                                    if u.get_cargo_space_left() < target_unit.get_cargo_space_left():
                                        # This new target it better, it has less space left and can take all our
                                        # resources
                                        target_unit = u
                                    
                                elif( target_unit.get_cargo_space_left() >= resource_amount ):
                                    # Don't change targets. Current one is best since it can take all
                                    # the resources, but new target can't.
                                    pass
                                    
                                elif( u.get_cargo_space_left() > target_unit.get_cargo_space_left() ):
                                    # Change targets, because neither target can accept all our resources and 
                                    # this target can take more resources.
                                    target_unit = u
                            elif u.type == Constants.UNIT_TYPES.CART:
                                # Transfer to this cart instead of the current worker target
                                target_unit = u
    
    # Build the transfer action request
    target_unit_id = None
    if target_unit is not None:
        target_unit_id = target_unit.id

        # Update the transfer amount based on the room of the target
        if target_unit.get_cargo_space_left() < resource_amount:
            resource_amount = target_unit.get_cargo_space_left()
    
    return TransferAction(team, unit_id, target_unit_id, resource_type, resource_amount)

########################################################################################################################
# This is the Agent that you need to design for the competition
########################################################################################################################
class AgentPolicy(Agent):
    def __init__(self, mode="train", model=None) -> None:
        """
        Arguments:
            mode: "train" or "inference", which controls if this agent is for training or not.
            model: The pretrained model, or if None it will operate in training mode.
        """
        super().__init__()
        self.model = model
        self.mode = mode
        self.stats = None
        self.stats_last_game = None
        
        self.map_layers = {
            Constants.UNIT_TYPES.CART + OFFSET_OPPONENT: 0,
            Constants.UNIT_TYPES.WORKER + OFFSET_OPPONENT: 1,
            CITIES + OFFSET_OPPONENT: 2,
            Constants.UNIT_TYPES.CART: 3,
            Constants.UNIT_TYPES.WORKER: 4,
            CITIES: 5,
            "UnitCargoWood": 6,
            "UnitCargoCoal": 7,
            "UnitCargoUranium": 8,
            "UnitOrCityFuelValue": 9,
            "Cooldown": 10,
            "RoadLevel": 11,
            Constants.RESOURCE_TYPES.WOOD: 12,
            Constants.RESOURCE_TYPES.COAL: 13,
            Constants.RESOURCE_TYPES.URANIUM: 14,
            "ActiveObjectPos": 15,
            "ActiveObjectGradX": 16,
            "ActiveObjectGradY": 17,
            "ActiveObjectDist": 18,
            # "ActiveObjectIsWorker": 19,
            # "ActiveObjectIsCart": 20,
            # "ActiveObjectIsCityTile": 21,
            "GameTimeToNight" : 19,
            "GameTimeToDay" : 20,
            "GamePercentDone" : 21,
            "GameResearch" : 22,
            "GameResearchCoal" : 23,
            "GameResearchUranium" : 24,
        }

        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.actionSpaceMapUnit = [
            SpawnCityAction,
            # smart_transfer_to_nearby,
            # partial(MoveAction, direction=Constants.DIRECTIONS.CENTER),  # This is the do-nothing action
            partial(MoveAction, direction=Constants.DIRECTIONS.NORTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.WEST),
            partial(MoveAction, direction=Constants.DIRECTIONS.SOUTH),
            partial(MoveAction, direction=Constants.DIRECTIONS.EAST),
            
            #PillageAction,
        ]
        
        self.actionSpaceMapCity = [
            SpawnWorkerAction,
            SpawnCartAction,
            ResearchAction,
            ]
        
        self.action_space = spaces.Discrete(len(self.actionSpaceMapUnit))

        # Observation space: (Basic minimum for a miner agent)
        # Object:
        #   1x is worker
        #   1x is cart
        #   1x is citytile
        #
        #   5x direction_nearest_wood
        #   1x distance_nearest_wood
        #   1x amount
        #
        #   5x direction_nearest_coal
        #   1x distance_nearest_coal
        #   1x amount
        #
        #   5x direction_nearest_uranium
        #   1x distance_nearest_uranium
        #   1x amount
        #
        #   5x direction_nearest_city
        #   1x distance_nearest_city
        #   1x amount of fuel
        #
        #   28x (the same as above, but direction, distance, and amount to the furthest of each)
        #
        #   5x direction_nearest_worker
        #   1x distance_nearest_worker
        #   1x amount of cargo
        # Unit:
        #   1x cargo size
        # State:
        #   1x is night
        #   1x percent of game done
        #   2x citytile counts [cur player, opponent]
        #   2x worker counts [cur player, opponent]
        #   2x cart counts [cur player, opponent]
        #   1x research points [cur player]
        #   1x researched coal [cur player]
        #   1x researched uranium [cur player]
        self.observation_shape = (3 + 7 * 5 * 2 + 1 + 1 + 1 + 2 + 2 + 2 + 3,)
        # self.observation_space = spaces.Box(low=0, high=1, shape=
        # self.observation_shape, dtype=np.float16)
        
        self.observation_space_dict = {
            # Map of the game, organize as maximum map size (num_layersx32x32)
            "map": spaces.Box(low=-1, high=1, shape=(len(self.map_layers), 32, 32), dtype=np.float16), # 32x32x13 = 13,312
            "unit_info": spaces.Box(low=-1, high=1, shape=(3 + 7 * 5 * 2 + 1 + 1 + 1 + 2 + 2 + 2 + 3,), dtype=np.float16), # 32x32x13 = 13,312
            "size": spaces.Box(low=1, high=32, shape=(2,), dtype=np.int32),
            "pos": spaces.Box(low=0, high=32, shape=(2,), dtype=np.int32),
        }
        
        self.observation_space = spaces.Dict(self.observation_space_dict)

        self.object_nodes = {}

    def get_agent_type(self):
        """
        Returns the type of agent. Use AGENT for inference, and LEARNING for training a model.
        """
        if self.mode == "train":
            return Constants.AGENT_TYPE.LEARNING
        else:
            return Constants.AGENT_TYPE.AGENT

    def get_observation(self, game, unit, city_tile, team, is_new_turn):
        """
        Implements getting a observation from the current game for this unit or city
        """
        # Map layer definition
        unit_type_string_map = {
            Constants.UNIT_TYPES.WORKER: "WORKER",
            Constants.UNIT_TYPES.CART: "CART",
        }
        
        if is_new_turn:
            # It's a new turn this event. This flag is set True for only the first observation from each turn.
            # Update any per-turn fixed observation space that doesn't change per unit/city controlled.
            self.obs = OrderedDict(
                [
                    ("map", np.zeros((len(self.map_layers),32,32))),
                    ("size", np.zeros((2))),
                    ("unit_info", np.zeros((3 + 7 * 5 * 2 + 1 + 1 + 1 + 2 + 2 + 2 + 3,)))
                ]
            )
            
            # Update night and day states
            day_length = game.configs["parameters"]["DAY_LENGTH"]
            night_length = game.configs["parameters"]["NIGHT_LENGTH"]
            cycle_length = day_length + night_length
            self.obs["map"][self.map_layers["GamePercentDone"],:,:] = np.ones((32,32)) * (game.state["turn"] / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"])
            if game.is_night():
                self.obs["map"][self.map_layers["GameTimeToNight"],:,:] = np.zeros((32,32))
                self.obs["map"][self.map_layers["GameTimeToDay"],:,:] = np.ones((32,32)) * (((game.state["turn"] % cycle_length) - day_length) / night_length)
            else:
                self.obs["map"][self.map_layers["GameTimeToNight"],:,:] = np.ones((32,32)) * (game.state["turn"] % cycle_length) / day_length
                self.obs["map"][self.map_layers["GameTimeToDay"],:,:] = np.zeros((32,32))

            # Update research state
            self.obs["map"][self.map_layers["GameResearch"],:,:] = np.ones((32,32)) * game.state["teamStates"][team]["researchPoints"] / 200.0
            self.obs["map"][self.map_layers["GameResearchCoal"],:,:] = np.ones((32,32)) * float(game.state["teamStates"][team]["researched"]["coal"])
            self.obs["map"][self.map_layers["GameResearchUranium"],:,:] = np.ones((32,32)) * float(game.state["teamStates"][team]["researched"]["uranium"])
            
            # Populate unit layer
            for t in [team, (team + 1) % 2]:
                for u in game.state["teamStates"][t]["units"].values():
                    layer = u.type
                    if u.team != team:
                        layer += OFFSET_OPPONENT
                    
                    if layer in self.map_layers:
                        # Add 0.25 for each unit on each spot. This allows for up to 4 overlapping units of the same time on the same tile (on citytiles)
                        value = min(
                            self.obs["map"][self.map_layers[layer]][u.pos.x][u.pos.y] + 0.25,
                            1.0
                        )
                        self.obs["map"][self.map_layers[layer]][u.pos.x][u.pos.y] = value

                    # Unit storage of each resource as % of total
                    self.obs["map"][self.map_layers["UnitCargoWood"]][u.pos.x][u.pos.y] = (
                        u.cargo["wood"] / GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"][unit_type_string_map[u.type]]
                    )
                    self.obs["map"][self.map_layers["UnitCargoCoal"]][u.pos.x][u.pos.y] = (
                        u.cargo["coal"] / GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"][unit_type_string_map[u.type]]
                    )
                    self.obs["map"][self.map_layers["UnitCargoUranium"]][u.pos.x][u.pos.y] = (
                        u.cargo["uranium"] / GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"][unit_type_string_map[u.type]]
                    )

                    # Unit fuel value, overriden by city tiles later
                    self.obs["map"][self.map_layers["UnitOrCityFuelValue"]][u.pos.x][u.pos.y] = min(
                        u.get_cargo_fuel_value() / (GAME_CONSTANTS["PARAMETERS"]["LIGHT_UPKEEP"]["CITY"] * 100 ),
                        1.0
                    )

                    # Set cooldown, will be overriden by city tile cooldowns
                    self.obs["map"][self.map_layers["Cooldown"]][u.pos.x][u.pos.y] = (
                        u.cooldown / GAME_CONSTANTS["PARAMETERS"]["UNIT_ACTION_COOLDOWN"][unit_type_string_map[u.type]]
                    )

            
            
            
            # Populate city layer
            for c in game.cities.values():
                # Load the layer for this city
                layer = CITIES
                if c.team != t:
                    layer += OFFSET_OPPONENT

                # Iterate the citytiles
                for cc in c.city_cells:
                    ct = cc.city_tile

                    # Mark this city tile location
                    if layer in self.map_layers:
                        self.obs["map"][self.map_layers[layer]][cc.pos.x][cc.pos.y] = 1 # Mark a city tile here

                    # City fuel value, overides unit fuel
                    # City fuel as % of upkeep for 100 turns
                    if c.team == t:
                        try:
                            self.obs["map"][self.map_layers["UnitOrCityFuelValue"]][cc.pos.x][cc.pos.y] = min(
                                c.fuel / (c.get_light_upkeep()),
                                1.0
                            )
                        except:
                            pass
                        
                        # Set citytile cooldown, overrides unit cooldowns on same tile
                        self.obs["map"][self.map_layers["Cooldown"]][cc.pos.x][cc.pos.y] = ct.cooldown / GAME_CONSTANTS["PARAMETERS"]["CITY_ACTION_COOLDOWN"]

            # Populate resources layer
            scaling_by_resource = {
                Constants.RESOURCE_TYPES.WOOD: 800,
                Constants.RESOURCE_TYPES.COAL: 300,
                Constants.RESOURCE_TYPES.URANIUM: 200,
            }
            for c in game.map.resources:
                if c.resource.type in self.map_layers:
                    self.obs["map"][self.map_layers[c.resource.type]][c.pos.x][c.pos.y] = min(
                        c.resource.amount / scaling_by_resource[c.resource.type],
                        1.0
                    )
            
            # Populate the road layer
            # for c in game.cells_with_roads:
            #     self.obs["map"][self.map_layers["RoadLevel"]][c.pos.x][c.pos.y] = min(
            #             c.road / game.configs["parameters"]["MAX_ROAD"],
            #             1.0
            #         )
        
        # Helper variables
        active = None
        city = None
        if city_tile != None:
            active = city_tile
            city = game.cities[city_tile.city_id]
        unit_type_string = None
        
        if unit != None:
            active = unit
            unit_type_string = unit_type_string_map[unit.type]
        
        
        # Fill in the current actioning unit type
        # self.obs["map"][self.map_layers["ActiveObjectIsWorker"],:,:] = np.zeros((32,32))
        # self.obs["map"][self.map_layers["ActiveObjectIsCart"],:,:] = np.zeros((32,32))
        # self.obs["map"][self.map_layers["ActiveObjectIsCityTile"],:,:] = np.zeros((32,32))

        # type = None
        # if unit is not None:
        #     if unit.type == Constants.UNIT_TYPES.WORKER:
        #         # Worker
        #         self.obs["map"][self.map_layers["ActiveObjectIsWorker"],:,:] = np.ones((32,32))
        #     else:
        #         self.obs["map"][self.map_layers["ActiveObjectIsCart"],:,:] = np.ones((32,32))
        # if city_tile is not None:
        #     self.obs["map"][self.map_layers["ActiveObjectIsCityTile"],:,:] = np.ones((32,32))
        
        
        
        # Active object position and gradients
        self.obs["map"][self.map_layers["ActiveObjectPos"],active.pos.x,active.pos.y] = 1.0
        x, y = np.ogrid[0:32, 0:32]
        self.obs["map"][self.map_layers["ActiveObjectGradX"],:,:] = (x - active.pos.x) / 32
        self.obs["map"][self.map_layers["ActiveObjectGradY"],:,:] = (y - active.pos.y) / 32
        self.obs["map"][self.map_layers["ActiveObjectDist"],:,:] = (np.abs(y - active.pos.y) + np.abs(x - active.pos.x)) / 32
        
        
        self.obs["size"] = game.map.height, game.map.width
        self.obs["pos"] = active.pos.x, active.pos.y
        
        
        observation_index = 0
        
        # now_map = np.zeros((7, game.map.height, game.map.width))
        
        # print('dir(game.map):', dir(game.map))
        # print('game.map.resources[0].resource.type:', game.map.resources[0].resource.type)
        
        if is_new_turn:
            # It's a new turn this event. This flag is set True for only the first observation from each turn.
            # Update any per-turn fixed observation space that doesn't change per unit/city controlled.

            # Build a list of object nodes by type for quick distance-searches
            self.object_nodes = {}

            # Add resources
            for cell in game.map.resources:
                if cell.resource.type not in self.object_nodes:
                    self.object_nodes[cell.resource.type] = np.array([[cell.pos.x, cell.pos.y]])
                else:
                    self.object_nodes[cell.resource.type] = np.concatenate(
                        (
                            self.object_nodes[cell.resource.type],
                            [[cell.pos.x, cell.pos.y]]
                        ),
                        axis=0
                    )
                

            # Add your own and opponent units
            for t in [team, (team + 1) % 2]:
                for u in game.state["teamStates"][team]["units"].values():
                    key = str(u.type)
                    if t != team:
                        key = str(u.type) + "_opponent"
                        

                    if key not in self.object_nodes:
                        self.object_nodes[key] = np.array([[u.pos.x, u.pos.y]])
                    else:
                        self.object_nodes[key] = np.concatenate(
                            (
                                self.object_nodes[key],
                                [[u.pos.x, u.pos.y]]
                            )
                            , axis=0
                        )

            # Add your own and opponent cities
            for city in game.cities.values():
                for cells in city.city_cells:
                    key = "city"
                    c = game.cities[cells.city_tile.city_id]
                    if city.team != team:
                        key = "city_opponent"
                        

                    if key not in self.object_nodes:
                        self.object_nodes[key] = np.array([[cells.pos.x, cells.pos.y]])
                    else:
                        self.object_nodes[key] = np.concatenate(
                            (
                                self.object_nodes[key],
                                [[cells.pos.x, cells.pos.y]]
                            )
                            , axis=0
                        )
        
        
        
        # Observation space: (Basic minimum for a miner agent)
        # Object:
        #   1x is worker
        #   1x is cart
        #   1x is citytile
        #   5x direction_nearest_wood
        #   1x distance_nearest_wood
        #   1x amount
        #
        #   5x direction_nearest_coal
        #   1x distance_nearest_coal
        #   1x amount
        #
        #   5x direction_nearest_uranium
        #   1x distance_nearest_uranium
        #   1x amount
        #
        #   5x direction_nearest_city
        #   1x distance_nearest_city
        #   1x amount of fuel
        #
        #   5x direction_nearest_worker
        #   1x distance_nearest_worker
        #   1x amount of cargo
        #
        #   28x (the same as above, but direction, distance, and amount to the furthest of each)
        #
        # Unit:
        #   1x cargo size
        # State:
        #   1x is night
        #   1x percent of game done
        #   2x citytile counts [cur player, opponent]
        #   2x worker counts [cur player, opponent]
        #   2x cart counts [cur player, opponent]
        #   1x research points [cur player]
        #   1x researched coal [cur player]
        #   1x researched uranium [cur player]
        self.obs['unit_info'] = np.zeros(self.observation_shape)
        
        # Update the type of this object
        #   1x is worker
        #   1x is cart
        #   1x is citytile
        observation_index = 0
        if unit is not None:
            if unit.type == Constants.UNIT_TYPES.WORKER:
                self.obs['unit_info'][observation_index : observation_index+1] = 1.0 # Worker
            else:
                self.obs['unit_info'][observation_index+1 : observation_index+2] = 1.0 # Cart
        if city_tile is not None:
            self.obs['unit_info'][observation_index+2 : observation_index+3] = 1.0 # CityTile
        observation_index += 3
        
        pos = None
        if unit is not None:
            pos = unit.pos
        else:
            pos = city_tile.pos

        if pos is None:
            observation_index += 7 * 5 * 2
        else:
            # Encode the direction to the nearest objects
            #   5x direction_nearest
            #   1x distance
            for distance_function in [closest_node, furthest_node]:
                for key in [
                    Constants.RESOURCE_TYPES.WOOD,
                    Constants.RESOURCE_TYPES.COAL,
                    Constants.RESOURCE_TYPES.URANIUM,
                    "city",
                    str(Constants.UNIT_TYPES.WORKER)]:
                    # Process the direction to and distance to this object type

                    # Encode the direction to the nearest object (excluding itself)
                    #   5x direction
                    #   1x distance
                    if key in self.object_nodes:
                        if (
                                (key == "city" and city_tile is not None) or
                                (unit is not None and unit.type == key and len(get_cell_by_pos(unit.pos).units) <= 1 )
                        ):
                            # Filter out the current unit from the closest-search
                            closest_index = closest_node((pos.x, pos.y), self.object_nodes[key])
                            filtered_nodes = np.delete(self.object_nodes[key], closest_index, axis=0)
                        else:
                            filtered_nodes = self.object_nodes[key]

                        if len(filtered_nodes) == 0:
                            # No other object of this type
                            self.obs['unit_info'][observation_index + 5] = 1.0
                        else:
                            # There is another object of this type
                            closest_index = distance_function((pos.x, pos.y), filtered_nodes)

                            if closest_index is not None and closest_index >= 0:
                                closest = filtered_nodes[closest_index]
                                closest_position = Position(closest[0], closest[1])
                                direction = pos.direction_to(closest_position)
                                mapping = {
                                    Constants.DIRECTIONS.CENTER: 0,
                                    Constants.DIRECTIONS.NORTH: 1,
                                    Constants.DIRECTIONS.WEST: 2,
                                    Constants.DIRECTIONS.SOUTH: 3,
                                    Constants.DIRECTIONS.EAST: 4,
                                }
                                self.obs['unit_info'][observation_index + mapping[direction]] = 1.0  # One-hot encoding direction

                                # 0 to 1 distance
                                distance = pos.distance_to(closest_position)
                                self.obs['unit_info'][observation_index + 5] = min(distance / 20.0, 1.0)

                                # 0 to 1 value (amount of resource, cargo for unit, or fuel for city)
                                if key == "city":
                                    # City fuel as % of upkeep for 200 turns
                                    c = game.cities[game.map.get_cell_by_pos(closest_position).city_tile.city_id]
                                    self.obs['unit_info'][observation_index + 6] = min(
                                        c.fuel / (c.get_light_upkeep() * 200.0),
                                        1.0
                                    )
                                elif key in [Constants.RESOURCE_TYPES.WOOD, Constants.RESOURCE_TYPES.COAL,
                                             Constants.RESOURCE_TYPES.URANIUM]:
                                    # Resource amount
                                    self.obs['unit_info'][observation_index + 6] = min(
                                        game.map.get_cell_by_pos(closest_position).resource.amount / 500,
                                        1.0
                                    )
                                else:
                                    # Unit cargo
                                    self.obs['unit_info'][observation_index + 6] = min(
                                        next(iter(game.map.get_cell_by_pos(
                                            closest_position).units.values())).get_cargo_space_left() / 100,
                                        1.0
                                    )

                    observation_index += 7

        if unit is not None:
            # Encode the cargo space
            #   1x cargo size
            self.obs['unit_info'][observation_index] = unit.get_cargo_space_left() / GAME_CONSTANTS["PARAMETERS"]["RESOURCE_CAPACITY"][
                "WORKER"]
            observation_index += 1
        else:
            observation_index += 1

        # Game state observations

        #   1x is night
        self.obs['unit_info'][observation_index] = game.is_night()
        observation_index += 1

        #   1x percent of game done
        self.obs['unit_info'][observation_index] = game.state["turn"] / GAME_CONSTANTS["PARAMETERS"]["MAX_DAYS"]
        observation_index += 1

        #   2x citytile counts [cur player, opponent]
        #   2x worker counts [cur player, opponent]
        #   2x cart counts [cur player, opponent]
        max_count = 30
        for key in ["city", str(Constants.UNIT_TYPES.WORKER), str(Constants.UNIT_TYPES.CART)]:
            if key in self.object_nodes:
                self.obs['unit_info'][observation_index] = len(self.object_nodes[key]) / max_count
            if (key + "_opponent") in self.object_nodes:
                self.obs['unit_info'][observation_index + 1] = len(self.object_nodes[(key + "_opponent")]) / max_count
            observation_index += 2

        #   1x research points [cur player]
        #   1x researched coal [cur player]
        #   1x researched uranium [cur player]
        self.obs['unit_info'][observation_index] = game.state["teamStates"][team]["researchPoints"] / 200.0
        self.obs['unit_info'][observation_index+1] = float(game.state["teamStates"][team]["researched"]["coal"])
        self.obs['unit_info'][observation_index+2] = float(game.state["teamStates"][team]["researched"]["uranium"])
        
        
        
        
        # return {'state': obs.float(), 'map': now_map.float()}
        # print('observation:', self.obs)
        return self.obs

    def action_code_to_action(self, action_code, game, unit=None, city_tile=None, team=None):
        """
        Takes an action in the environment according to actionCode:
            actionCode: Index of action to take into the action array.
        Returns: An action.
        """
        # Map actionCode index into to a constructed Action object
        try:
            x = None
            y = None
            if city_tile is not None:
                x = city_tile.pos.x
                y = city_tile.pos.y
            elif unit is not None:
                x = unit.pos.x
                y = unit.pos.y

            if city_tile is None:
                return self.actionSpaceMapUnit[action_code](
                    game=game,
                    unit_id=unit.id if unit else None,
                    unit=unit,
                    city_id=city_tile.city_id if city_tile else None,
                    citytile=city_tile,
                    team=team,
                    x=x,
                    y=y
                )
            else:
                action_code = 0
                action = self.actionSpaceMapCity[action_code](
                    game=game,
                    unit_id=unit.id if unit else None,
                    unit=unit,
                    city_id=city_tile.city_id if city_tile else None,
                    citytile=city_tile,
                    team=team,
                    x=x,
                    y=y
                )
                if not action.is_valid(game, actions_validated=[]):
                    # if action_code == 0:
                    #     print('invalid spawn worker')
                    action = ResearchAction(
                        game=game,
                        unit_id=unit.id if unit else None,
                        unit=unit,
                        city_id=city_tile.city_id if city_tile else None,
                        citytile=city_tile,
                        team=team,
                        x=x,
                        y=y
                    )
                return action
        except Exception as e:
            # Not a valid action
            print(e)
            return None

    def take_action(self, action_code, game, unit=None, city_tile=None, team=None):
        """
        Takes an action in the environment according to actionCode:
            actionCode: Index of action to take into the action array.
        """
        action = self.action_code_to_action(action_code, game, unit, city_tile, team)
        self.match_controller.take_action(action)
    
    def game_start(self, game):
        """
        This funciton is called at the start of each game. Use this to
        reset and initialize per game. Note that self.team may have
        been changed since last game. The game map has been created
        and starting units placed.

        Args:
            game ([type]): Game.
        """

        self.last_fuel = 0


    def get_reward(self, game, is_game_finished, is_new_turn, is_game_error):
        """
        Returns the reward function for this step of the game.
        """
        if is_game_error:
            # Game environment step failed, assign a game lost reward to not incentivise this
            print("Game failed due to error")
            return -1.0

        if not is_new_turn and not is_game_finished:
            # Only apply rewards at the start of each turn
            return 0
        

        # Get some basic stats
        unit_count = len(game.state["teamStates"][self.team % 2]["units"])
        unit_count_opponent = len(game.state["teamStates"][(self.team + 1) % 2]["units"])
        research_points = min(game.state["teamStates"][self.team]["researchPoints"], 200.0) # Cap research points at 200
        city_count = 0
        city_count_opponent = 0
        city_tile_count = 0
        city_tile_count_opponent = 0
        city_tile_all = 0
        for city in game.cities.values():
            if city.team == self.team:
                city_count += 1
            else:
                city_count_opponent += 1

            for cell in city.city_cells:
                if city.team == self.team:
                    city_tile_all += 1
                    if city.fuel > city.get_light_upkeep():
                        city_tile_count += 1
                else:
                    city_tile_count_opponent += 1
        
        fuel_count = 0         
        for u in game.state["teamStates"][self.team]["units"].values():
            fuel_count += u.cargo["coal"]*2 + u.cargo["uranium"]*10
        
        new_fuel = abs(fuel_count - self.last_fuel)
        self.last_fuel = fuel_count

        # Give a reward each turn for each tile and unit alive each turn
        #reward_state = city_tile_count * 0.01 + unit_count * 0.001 + research_points * 0.00001
        reward_state = city_tile_count * 0.1 + city_tile_all * 0.03 + new_fuel * 0.01 + unit_count * 0.01
        
        if is_game_finished:
            print('turn:', game.state['turn'])
            if game.state['turn'] == 360:
                print('360 turns reward, left citytiles: {}, left workers: {}'.format(city_tile_all, unit_count))
            # Give a bigger reward for end-of-game unit and city count
            if game.get_winning_team() == self.team:
                # print("Won game. %i cities, %i citytiles, %i units." % (cityCount, cityTileCount, unitCount))
                return city_tile_all * 8 * game.state['turn'] / 360
            else:
                # print("Lost game. %i cities, %i citytiles, %i units." % (cityCount, cityTileCount, unitCount))
                return city_tile_all * 3 * game.state['turn'] / 360
        else:
            # Calculate the current reward state
            # If you want, any micro rewards or other rewards that are not win/lose end-of-game rewards
            # As unit count increases, loss automatically decreases without this compensation because there are more
            # steps per turn, and one reward per turn.
            return reward_state

    def process_turn(self, game, team):
        """
        Decides on a set of actions for the current turn. Not used in training, only inference.
        Returns: Array of actions to perform.
        """
        start_time = time.time()
        actions = []
        new_turn = True

        # Inference the model per-unit
        units = game.state["teamStates"][team]["units"].values()
        for unit in units:
            if unit.can_act():
                obs = self.get_observation(game, unit, None, unit.team, new_turn)
                action_code, _states = self.model.predict(obs, deterministic=True)
                # print('action_code:', action_code)
                if action_code is not None:
                    actions.append(
                        self.action_code_to_action(action_code, game=game, unit=unit, city_tile=None, team=unit.team))
                new_turn = False

        # Inference the model per-city
        cities = game.cities.values()
        for city in cities:
            if city.team == team:
                for cell in city.city_cells:
                    city_tile = cell.city_tile
                    if city_tile.can_act():
                        obs = self.get_observation(game, None, city_tile, city.team, new_turn)
                        action_code, _states = self.model.predict(obs, deterministic=True)
                        if action_code is not None:
                            actions.append(
                                self.action_code_to_action(action_code, game=game, unit=None, city_tile=city_tile,
                                                           team=city.team))
                        new_turn = False

        time_taken = time.time() - start_time
        if time_taken > 0.5:  # Warn if larger than 0.5 seconds.
            print("WARNING: Inference took %.3f seconds for computing actions. Limit is 1 second." % time_taken,
                  file=sys.stderr)

        return actions

