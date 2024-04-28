from abc import ABC, abstractmethod
from typing import Tuple, List, FrozenSet
from dataclasses import dataclass
import random


class BattleshipException(Exception):
    """Base Class for Battleship Exceptions"""


class InvalidShipCoordinateException(BattleshipException):
    """Raises error when a ship's coordinates are invalid."""


class InvalidHitMoveException(BattleshipException):
    """Raises error when a player tries to give an invalid hit move coordinate."""


class BoardException(BattleshipException):
    pass


class GameException(BattleshipException):
    pass


class Ship:
    def __init__(self, coordinates: List[Tuple[int, int]]):
        self.coordinates = coordinates
        self._check_ship_coordinates()

        self._un_hit_coordinates: FrozenSet[Tuple[int, int]] = {tup for tup in self.coordinates}

    @property
    def is_destroyed(self) -> bool:
        return len(self._un_hit_coordinates) == 0

    @property
    def num_hits(self) -> int:
        # return len(self.coordinates) - len(self._un_hit_coordinates)
        return len(self.hit_cells)

    @property
    def hit_cells(self) -> List[Tuple[int, int]]:
        return list(set(self.coordinates) - self._un_hit_coordinates)

    @property
    def un_hit_cells(self) -> List[Tuple[int, int]]:
        return list(self._un_hit_coordinates)

    def hit_ship(self, row, col) -> None:
        if self.is_destroyed:
            raise InvalidHitMoveException("The ship is already destroyed.")
        if (row, col) not in self._un_hit_coordinates:
            raise InvalidHitMoveException(f"{(row, col)} is an invalid hit move coordinate.")

        self._un_hit_coordinates.discard((row, col))

    def is_ship_overlap(self, other) -> bool:
        if type(other) is not self.__class__:
            raise ValueError(f"The given ship is not of type `{self.__class__.__name__}`")

        return any([tup in other.coordinates for tup in self.coordinates])

    def _check_ship_coordinates(self) -> None:
        """Check if the ship's coordinates are valid."""
        rows = sorted([tup[0] for tup in self.coordinates])
        columns = sorted([tup[1] for tup in self.coordinates])

        # Check if there are given coordinates for ship's position
        if len(self.coordinates) < 1:
            raise InvalidShipCoordinateException("Cannot instantiate a ship without coordinates for it's position.")

        # Check if any of rows or columns are less than 0
        if any(row < 0 for row in rows):
            raise InvalidShipCoordinateException("One of the ship's coordinates have negative row value.")
        if any(col < 0 for col in columns):
            raise InvalidShipCoordinateException("One of the ship's coordinates have negative column value.")

        if len(self.coordinates) > 1:
            # Check if neither the rows or columns have constant value
            is_row_constant = not any([rows[j] != rows[j + 1] for j in range(len(rows) - 1)])
            is_col_constant = not any([columns[i] != columns[i + 1] for i in range(len(columns) - 1)])

            if any([rows[j] != rows[j + 1] for j in range(len(rows) - 1)]) and any(
                    [columns[i] != columns[i + 1] for i in range(len(columns) - 1)]):
                raise InvalidShipCoordinateException("Neither the rows or columns have constant value.")

            # Check if there's a duplicate tuple in the coordindates
            if len(self.coordinates) != len(set(self.coordinates)):
                raise InvalidShipCoordinateException("There is a duplicate in one of the ship's coordinates.")

            # Check if the non-constant row/column is in consecutive order
            if is_row_constant:
                # The columns must be consecutive
                assert all([columns[idx] + 1 == columns[idx + 1] for idx in
                            range(len(columns) - 1)]), "The columns are not in consecutive order."
            if is_col_constant:
                # The rows must be consecutive
                assert all([rows[idx] + 1 == rows[idx + 1] for idx in
                            range(len(rows) - 1)]), "The rows are not in consecutive order."


def generate_row_ship_cells(board_size: int):
    for ship_size in range(1, board_size + 1):
        for i in range(board_size):  # Rows
            for j in range(board_size - ship_size + 1):  # Columns
                coordinates = [(i, col) for col in range(j, j + ship_size)]
                yield coordinates


def generate_column_ship_cells(board_size: int):
    for ship_size in range(1, board_size + 1):
        for j in range(board_size):  # Columns
            for i in range(board_size - ship_size + 1):  # Rows
                coordinates = [(row, j) for row in range(i, i + ship_size)]
                yield coordinates


def generate_random_row_ship(board_size: int, ship_length: int) -> Ship:
    row_ships = [Ship(cells) for cells in generate_row_ship_cells(board_size) if len(cells) == ship_length]
    return random.choice(row_ships)


def generate_random_column_ship(board_size: int, ship_length: int) -> Ship:
    column_ships = [Ship(cells) for cells in generate_column_ship_cells(board_size) if len(cells) == ship_length]
    return random.choice(column_ships)


def generate_random_ships_arrangements(board_size: int) -> List[Ship]:
    ships = []

    i = board_size
    while i > 0:
        row_or_col = random.choice(['row', 'column'])
        if row_or_col == 'row':
            ship = generate_random_row_ship(board_size, i)
        else:
            ship = generate_random_column_ship(board_size, i)

        if any(ship.is_ship_overlap(s) for s in ships):
            continue

        ships.append(ship)
        i -= 1

    n = len(ships)
    assert (n * (n + 1)) / 2 == len(
        [tup for ship in ships for tup in ship.coordinates]), "Invalid Mathematical Assumption"

    return ships


@dataclass
class BoardStatesPlayerPOV:
    """Cell's State Labels from Player's Point-of-View."""
    missed: str = "M"
    hit: str = "H"
    ship: str = "S"
    unoccupied: str = "-"


@dataclass
class BoardStatesEnemyPOV:
    """Cell's State Labels from Enemy's Point-of-View."""
    hit: str = "X"
    missed: str = "O"
    no_move: str = "-"


class Board:
    def __init__(self,
                 board_size: int,
                 empty_label=" "):

        if board_size < 5 or board_size > 15:
            raise BoardException("Invalid given board size. Board size must be 5 to 15.")

        self.board_size = board_size
        self._empty_label = empty_label

        self._board = [[self._empty_label for _ in range(self.board_size)] for _ in range(self.board_size)]

        # Instantiate Labels for Player and Enemy POVs
        self._player_pov_labels = BoardStatesPlayerPOV()
        self._enemy_pov_labels = BoardStatesEnemyPOV()

        self.ships: List[Ship] = []  # List of Ships

        # Keep track of all enemy actions/moves for "Missed" and "No Move".
        # Hit can be obtained from `hit_cells` property.
        self._enemy_moves = {
            "hit": [],
            "missed": []
        }

    @property
    def num_of_ships(self):
        return len(self.ships)

    @property
    def dead_ships(self) -> List[Ship]:
        return [ship for ship in self.ships if ship.is_destroyed]

    @property
    def hit_cells(self) -> List[Tuple[int, int]]:
        return [cell for ship in self.ships for cell in ship.hit_cells]

    @property
    def un_hit_cells(self) -> List[Tuple[int, int]]:
        return [cell for ship in self.ships for cell in ship.un_hit_cells]

    @property
    def occupied_cells(self) -> List[Tuple[int, int]]:
        return [cell for ship in self.ships for cell in ship.coordinates]

    @property
    def is_player_lost(self) -> bool:
        return all(ship.is_destroyed for ship in self.ships)

    @property
    def valid_moves(self) -> List[Tuple[int, int]]:
        all_cells_generator = ((i, j) for i in range(self.board_size) for j in range(self.board_size))
        return [cell for cell in all_cells_generator if
                cell not in self._enemy_moves["hit"] + self._enemy_moves["missed"]]

    def generate_valid_moves(self):
        board = self.get_board_for_enemy()
        for i in range(self.board_size):
            for j in range(self.board_size):
                if board[i][j] == self._enemy_pov_labels.no_move:
                    # yield board[i][j]
                    yield i, j

    def get_board_for_player(self) -> List[List[str]]:
        """Gets the Board States from Player's POV"""
        # Initialize an empty board.
        board = self._create_empty_board()

        # Fill with Ships' Positions
        for ship in self.ships:
            for row, col in ship.coordinates:
                board[row][col] = self._player_pov_labels.ship

        # Fill with Hit
        for (row, col) in self._enemy_moves["hit"]:
            board[row][col] = self._player_pov_labels.hit

        # Fill with Missed
        for (row, col) in self._enemy_moves["missed"]:
            board[row][col] = self._player_pov_labels.missed

        # Fill with Unoccupied
        for row in range(self.board_size):
            for col in range(self.board_size):
                if board[row][col] == self._empty_label:
                    board[row][col] = self._player_pov_labels.unoccupied

        return board

    def get_board_for_enemy(self) -> List[List[str]]:
        """Gets the Board States from Enemy's POV"""
        # Initialize an empty board.
        board = self._create_empty_board()

        # Fill with Hit
        for row, col in self._enemy_moves["hit"]:
            board[row][col] = self._enemy_pov_labels.hit

        # Fill with Missed
        for row, col in self._enemy_moves["missed"]:
            board[row][col] = self._enemy_pov_labels.missed

        # Fill with No Moves
        for row in range(self.board_size):
            for col in range(self.board_size):
                if board[row][col] == self._empty_label:
                    board[row][col] = self._enemy_pov_labels.no_move

        return board

    def _create_empty_board(self) -> List[List[str]]:
        return [[self._empty_label for _ in range(self.board_size)] for _ in range(self.board_size)]

    @staticmethod
    def print_board(board: List[List[str]]) -> None:
        board_size = len(board)
        # Print the Column Numbers for first row (0 to board_size - 1)
        for col in range(board_size):
            print(f"   {col}", end="")
        print()

        for row in range(board_size):
            print(f"{row} ", end="")
            for col in range(board_size):
                print(f" {board[row][col]} ", end="")
                if col < board_size - 1:
                    print("|", end="")  # Separate cells with |
            print()  # Move to the next line after printing the row
        ...

    def place_ship(self, *coordinates: List[Tuple[int, int]]) -> None:
        """Place a ship on a board."""
        # Check if player can still place a ship based on the board & ship size constraints.
        if self.num_of_ships > self.board_size:
            raise BoardException("Cannot place another ship on the board.")

        # Check if there's an invalid cell/coordinate to place a ship.
        for row, col in coordinates:
            if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
                raise BoardException(f"Invalid cell to place a ship: ({row}, {col})")

        # Check if a cell is already occupied.
        if not all(not self._is_cell_occupied(*cell) for cell in coordinates):
            raise BoardException("Some of the given cells are already occupied.")

        # Create a New Ship
        new_ship = Ship(coordinates)

        # Add the ship to the list
        self.ships.append(new_ship)

    def _is_cell_occupied(self, row: int, col: int) -> bool:
        """Returns True if a cell is occupied by another ship. Otherwise False."""
        return any((row, col) in ship.coordinates for ship in self.ships)

    def which_ship(self, row: int, col: int) -> None | Ship:
        """Returns the Ship that occupies the cell. Otherwise None."""
        for ship in self.ships:
            if (row, col) in ship.coordinates:
                return ship
        return None

    def enemy_move(self, row: int, col: int) -> None:
        """Returns True if the given enemy's move hits a ship cell. Otherwise, False for missed."""
        ship: None | Ship = self.which_ship(row, col)

        if isinstance(ship, Ship):
            # Update states in the ship
            ship.hit_ship(row, col)
            self._enemy_moves["hit"].append((row, col))
        else:
            self._enemy_moves["missed"].append((row, col))


@dataclass
class Player:
    name: str
    enemy_board: Board

    def attack(self, row: int, col: int):
        self.enemy_board.enemy_move(row, col)

    def generate_random_ship_arrangements(self) -> None:
        ships = generate_random_ships_arrangements(self.enemy_board.board_size)
        for ship in ships:
            self.enemy_board.place_ship(*ship.coordinates)

    def place_ship(self, ship_coordinates):
        # Remind that this is for the enemy's ships not this player.
        self.enemy_board.place_ship(*ship_coordinates)


class RandomPlayer(Player):
    def random_attack(self):
        # tup = random.choice(self.enemy_board.valid_moves)  # Generate random coordinate
        tup = random.choice([(i, j) for i, j in self.enemy_board.generate_valid_moves()])  # Generate random coordinate
        self.attack(*tup)


class Game(ABC):
    MAX_BOARD_SIZE = 15
    MIN_BOARD_SIZE = 5

    def __init__(self,
                 board_size: int,
                 player_1_name: str,
                 player_2_name: str
                 ):

        # To-Do for Implementer:
        # - Initialize first player mover (i.e. self._current_player).
        # - Implement place_ships() and use it to place the ships on the Board for each player.
        # - Implement the game's main loop. This can be used for different UIs.

        self.player_1: Player = Player(player_1_name, Board(board_size))
        self.player_2: Player = Player(player_2_name, Board(board_size))
        self.board_size: int = board_size

        self._current_player: Player | None = None

    @property
    def current_player(self) -> Player | None:
        return self._current_player

    @current_player.setter
    def current_player(self, value):
        self._current_player = value

    @property
    def previous_player(self) -> Player | None:
        return self.player_1 if self._current_player.name == self.player_2.name else self.player_2

    def get_winner(self) -> Player | None:
        if self.player_1.enemy_board.is_player_lost:
            return self.player_1
        elif self.player_2.enemy_board.is_player_lost:
            return self.player_2
        else:
            return None

    def update_player(self) -> None:
        self._current_player = self.player_1 if self._current_player.name == self.player_2.name else self.player_2

    def make_current_player_attack(self, row: int, col: int) -> Player | None:
        if self.current_player is None:
            raise ValueError('First Player to move was not initialized.')

        # Make Current Player Attack. Note that exceptions will be thrown from .attack() method
        self.current_player.attack(row, col)

        # Check if there is a Winner
        winner_or_none = self.get_winner()
        if winner_or_none is not None:
            return winner_or_none

        # Update the Current Player
        self.update_player()

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    @abstractmethod
    def place_ships(self, *args, **kwargs):
        pass


class PromptMixin:
    @staticmethod
    def prompt_board_size(prompt_message: str,
                          error_message: str,
                          min_board_size: int = 5,
                          max_board_size: int = 15
                          ) -> int:

        while True:
            try:
                board_size = int(input(prompt_message))

                if board_size < min_board_size or board_size > max_board_size:
                    print(error_message)
                    continue

                return board_size

            except ValueError:
                print(error_message)
                continue

    @staticmethod
    def prompt_name(prompt_message: str, error_message: str = 'Invalid Input! Please try again.\n') -> str:
        while True:
            input_name = input(prompt_message)

            # Check for invalid user input
            if input_name.strip() == "":
                print(error_message)
                continue

            return input_name

    @staticmethod
    def boolean_prompt(prompt_message: str,
                       error_message: str = 'Invalid Input! Please try again.\n',
                       true_str: str = 'yes',
                       false_str: str = 'no') -> bool:
        while True:
            bool_inp = input(prompt_message)

            # Check for Invalid User Input
            if bool_inp.lower() not in [true_str.lower(), false_str.lower()]:
                print(error_message)
                continue

            return bool_inp.lower() == true_str.lower()

    @staticmethod
    def attack_prompt(board_size: int,
                      prompt_message: str,
                      error_message: str = 'Invalid Target! Pleas try again.',
                      ) -> Tuple[int, ...]:
        while True:
            attack_input = input(prompt_message)

            if len(attack_input.split()) != 2:
                print(error_message)
                continue

            try:
                attack_tuple = tuple(map(int, attack_input.split()))

                invalid_conditions = \
                    attack_tuple[0] < 0 or \
                    attack_tuple[0] >= board_size or \
                    attack_tuple[1] < 0 or \
                    attack_tuple[1] >= board_size

                if invalid_conditions:
                    print(f'Attack Coordinates must be in [0, {board_size})\n' +
                          error_message)
                    continue

                return attack_tuple
            except ValueError:
                print(error_message)
                continue


class PrintMixin:
    @staticmethod
    def print_player_board(player: Player, other_player: Player) -> None:
        player_board = other_player.enemy_board
        other_player_board = player.enemy_board

        player_board_state = player_board.get_board_for_player()
        player_hit_or_miss_state = other_player_board.get_board_for_enemy()

        print(f"{player.name} Battlefield Situation")
        Board.print_board(player_board_state)
        print()
        print(f"{player.name} Targets")
        Board.print_board(player_hit_or_miss_state)

        print("\n")


class UtilsMixin:
    @staticmethod
    def alternate_tuples(list1, list2):
        result = []
        max_len = max(len(list1), len(list2))

        for i in range(max_len):
            if i < len(list1):
                result.append(list1[i])
            if i < len(list2):
                result.append(list2[i])

        return result


class AlternatingGame(Game, UtilsMixin, PrintMixin):
    def shuffle(self) -> List[Tuple[int, int]]:
        # Given the board size, generate random unique sequence of move for one player until all board
        # cells are generated.

        out_list = [(i, j) for i in range(self.board_size) for j in range(self.board_size)]
        random.shuffle(out_list)  # In-place modification for shuffle
        return out_list

    def run(self,
            player_1_moves=None,
            player_2_moves=None,
            player_1_first=True
            ):

        # Use user given attack moves or generate random attack moves
        player_1_moves_ = player_1_moves or self.shuffle()
        player_2_moves_ = player_2_moves or self.shuffle()

        # Check if player 1 move first
        if player_1_first:
            self.current_player = self.player_1
            all_moves = self.alternate_tuples(player_1_moves_, player_2_moves_)
        else:

            all_moves = self.alternate_tuples(player_2_moves_, player_1_moves_)
            self.current_player = self.player_2

        print(f'{self.current_player.name} move first.')

        # Place the Ships on Boards
        self.place_ships()

        # Start Placing the alternative moves of the players
        for tup in all_moves:
            winner_or_none = self.make_current_player_attack(*tup)
            self.print_player_board(self.previous_player, self.current_player)

            if winner_or_none is not None:
                print(f'Player {winner_or_none.name} Won!!!')

                if winner_or_none == self.player_1:
                    self.print_player_board(self.player_1, self.player_2)
                else:
                    self.print_player_board(self.player_2, self.player_1)

                break

    def place_ships(self):
        # Use Random Positions for the Ships on the Board for each player
        self.player_1.generate_random_ship_arrangements()
        self.player_2.generate_random_ship_arrangements()


class CLIGameV2(Game, PrintMixin, PromptMixin):
    def __init__(self):
        board_size = self.prompt_board_size(f'Enter Board Size >>> ',
                                            'Invalid Board Size Input.\n')
        player_1_name = self.prompt_name("Please Enter the Name for Player 1 >>> ")
        self.play_with_random_player = self.boolean_prompt('Do you want to play with a bot? (yes/no) >>> ')
        player_2_name = self.prompt_name("Please Enter the Name for Player 2 >>> ")

        super().__init__(board_size, player_1_name, player_2_name)

        # Place the Ships Randomly
        self.place_ships()

        # Select the First Mover Randomly
        self.current_player = random.choice([self.player_1, self.player_2])

    def run(self):
        while True:
            try:
                # Play with a Bot
                if self.play_with_random_player:  # Assume that Player 2 is the Random Bot.
                    # Check if current player is player_2
                    if self.current_player == self.player_2:
                        # Make Random Valid Attack
                        valid_moves = [move for move in self.current_player.enemy_board.generate_valid_moves()]
                        attack_coordinate = random.choice(valid_moves)
                        winner_or_none = self.make_current_player_attack(*attack_coordinate)

                    # Player 1's move
                    else:
                        attack_coordinate = \
                            self.attack_prompt(self.board_size,
                                               f'Please Enter the Attack Coordinates Admiral {self.current_player.name} >>> ',
                                               'Invalid Attack Coordinate')
                        winner_or_none = self.make_current_player_attack(*attack_coordinate)

                        # Print Board State. When we make a move, it will automatically update for the next player
                        self.print_player_board(self.previous_player, self.current_player)
                else:
                    attack_coordinate = \
                        self.attack_prompt(self.board_size,
                                           f'Please Enter the Attack Coordinates Admiral {self.current_player.name} >>> ',
                                           'Invalid Attack Coordinate')

                    winner_or_none = self.make_current_player_attack(*attack_coordinate)

                    # Print Board State. When we make a move, it will automatically update for the next player
                    self.print_player_board(self.previous_player, self.current_player)

                if winner_or_none is not None:
                    print(f'Player {winner_or_none.name} won the war!!!')
                    break
            except Exception as e:
                print(e)
                continue

    def place_ships(self):
        self.player_1.generate_random_ship_arrangements()
        self.player_2.generate_random_ship_arrangements()


class CLIGame:
    MAX_BOARD_SIZE = 15
    MIN_BOARD_SIZE = 5

    def __init__(self):
        self._current_player = None
        self.player_1: Player | None = None
        self.player_2: Player | None = None
        self.board_size: int | None = None

        self.run()

    @property
    def current_player(self) -> Player | RandomPlayer | None:
        return self._current_player

    @property
    def previous_player(self) -> Player | RandomPlayer | None:
        return self.player_1 if self._current_player.name == self.player_2.name else self.player_2

    def update_player(self) -> None:
        self._current_player = self.player_1 if self._current_player.name == self.player_2.name else self.player_2

    def print_current_player_board(self) -> None:
        current_player_board = self.previous_player.enemy_board
        other_player_board = self.current_player.enemy_board

        current_player_board_state = current_player_board.get_board_for_player()
        current_player_hit_or_miss_state = other_player_board.get_board_for_enemy()

        print(f"{self.current_player.name} Battlefield Situation")
        Board.print_board(current_player_board_state)
        print()
        print(f"{self.current_player.name} Targets")
        Board.print_board(current_player_hit_or_miss_state)

        print("\n")

    @staticmethod
    def _prompt_name(prompt_message: str, error_message: str = 'Invalid Input! Please try again.\n') -> str:
        while True:
            input_name = input(prompt_message)

            # Check for invalid user input
            if input_name.strip() == "":
                print(error_message)
                continue

            return input_name

    @staticmethod
    def _boolean_prompt(prompt_message: str,
                        error_message: str = 'Invalid Input! Please try again.\n',
                        true_str: str = 'yes',
                        false_str: str = 'no') -> bool:
        while True:
            bool_inp = input(prompt_message)

            # Check for Invalid User Input
            if bool_inp.lower() not in [true_str.lower(), false_str.lower()]:
                print(error_message)
                continue

            return bool_inp.lower() == true_str.lower()

    def _attack_prompt(self,
                       prompt_message: str,
                       error_message: str = 'Invalid Target! Pleas try again.',
                       ) -> Tuple[int, ...]:
        while True:
            attack_input = input(prompt_message)

            if len(attack_input.split()) != 2:
                print(error_message)
                continue

            try:
                attack_tuple = tuple(map(int, attack_input.split()))

                invalid_conditions = \
                    attack_tuple[0] < 0 or \
                    attack_tuple[0] >= self.board_size or \
                    attack_tuple[1] < 0 or \
                    attack_tuple[1] >= self.board_size

                if invalid_conditions:
                    print(f'Attack Coordinates must be in [0, {self.board_size})\n' +
                          error_message)
                    continue

                return attack_tuple
            except ValueError:
                print(error_message)
                continue

    def get_winner(self) -> Player | RandomPlayer | None:
        if self.player_1.enemy_board.is_player_lost:
            return self.player_1
        elif self.player_2.enemy_board.is_player_lost:
            return self.player_2
        else:
            return None

    def run(self):
        print("======================== Welcome to Battleships ========================")
        print()

        # Prompt User for Board Size
        while True:
            try:
                self.board_size = int(input("Enter the Board Size (>= 5) >>> "))

                # Check given integer board size
                if self.board_size < self.MIN_BOARD_SIZE or self.board_size > self.MAX_BOARD_SIZE:
                    print("The Board Size must be in [5, 15]\n")
                    continue

                break
            except ValueError:
                print("Invalid Input!\n")
                continue

        player_1_name = self._prompt_name("Please Enter the Name for Player 1 >>> ")
        play_with_random_player = self._boolean_prompt('Do you want to play with a bot? (yes/no) >>> ')
        player_2_name = self._prompt_name("Please Enter the Name for Player 2 >>> ")

        # Instantiate Players
        self.player_1 = Player(player_1_name, Board(self.board_size))

        # Check if User wants to play with Random Bot
        if play_with_random_player:
            self.player_2 = RandomPlayer(player_2_name, Board(self.board_size))
        else:
            self.player_2 = Player(player_2_name, Board(self.board_size))

        # Generate Random Ship Arrange Method for both players
        self.player_1.generate_random_ship_arrangements()  # Arrangements for Player 2
        self.player_2.generate_random_ship_arrangements()  # Arrangements for Player 1

        # Randomly Select Initial Player to make a move
        self._current_player: Player = random.choice([self.player_1, self.player_2])

        # Run the Main Loop
        while True:
            # Current Player is not a Random Bot
            if not isinstance(self.current_player, RandomPlayer):
                try:
                    attack_tuple = self._attack_prompt(
                        f'Enter your next attack coordinate admiral {self.current_player.name} >>> ')

                    # Make the attack
                    self.current_player.attack(*attack_tuple)

                    # Print State for Current (Human) Player
                    self.print_current_player_board()
                except InvalidHitMoveException as e:
                    print(e)
                    continue
            else:
                assert isinstance(self.current_player, RandomPlayer), "Current Player is not a Random Player."
                print('Bot is attacking ...')
                self.current_player.random_attack()

            # Check if there is a winner.
            if isinstance(self.get_winner(), (Player, RandomPlayer)):
                break

            # Update the Current Player
            self.update_player()

        # Print the Results
        winner = self.get_winner()
        print(f'Player {winner.name} Wins!!!')


if __name__ == "__main__":
    # CLIGame()

    # game = CLIGameV2()
    # game.run()

    game = AlternatingGame(5, "Player 1", "Player 2")
    game.run(player_1_first=False)
