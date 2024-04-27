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
        self.num_ship = self.board_size - 1  # num_ships = board_size - 1
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
                    yield board[i][j]

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

        # Print rows
        for row in range(board_size):
            for col in range(board_size):
                print(f" {board[row][col]} ", end="")
                print("|", end="")  # Separate cells with |
            print()  # Move to the next line after printing the row

    def place_ship(self, *coordinates: List[Tuple[int, int]]) -> None:
        """Place a ship on a board."""
        # Check if player can still place a ship based on the board & ship size constraints.
        if len(self.ships) > self.board_size:
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

        if ship:
            # Update states in the ship
            ship.hit_ship(row, col)
            self._enemy_moves["hit"].append((row, col))
        else:
            self._enemy_moves["missed"].append((row, col))


def generate_random_attack_move(board: Board):
    return random.choice(board.valid_moves)


if __name__ == "__main__":
    pass
