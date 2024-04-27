from main import *
import unittest


class TestShip(unittest.TestCase):
    def test_valid_ship_instantiation(self):
        for cells in generate_row_ship_cells(5):
            ship = Ship(cells)
            self.assertFalse(ship.is_destroyed)
            self.assertEqual(ship.num_hits, 0)
        
        for cells in generate_column_ship_cells(5):
            ship = Ship(cells)
            self.assertFalse(ship.is_destroyed)
            self.assertEqual(ship.num_hits, 0)
    
    def test_invalid_ship_instantiation(self):
        ship_args = [[],
                    [(-1, 2), (51, 2), (2, 2)],
                    [(0, 2), (1, -12), (2, 2)],
                    [(0, 2), (1, 1), (2, 0)],
                    [(0, 2), (1, 2), (2, 2), (1, 2)]
                    ]
        
        for ship_arg in ship_args:
            with self.assertRaises(InvalidShipCoordinateException):
                Ship(ship_arg)
        
        ship_args = [
            [(0, 2), (1, 2), (4, 2)],
            [(1, 0), (1, 2), (1, 3)]
        ]
        
        for ship_arg in ship_args:
            with self.assertRaises(AssertionError):
                Ship(ship_arg)
    
    def test_ship_damaged_but_not_destroyed(self):
        ship = Ship([(0, 2), (1, 2), (2, 2)])
        ship.hit_ship(0, 2)
        ship.hit_ship(1, 2)
        
        self.assertEqual(ship.num_hits, 2)
        self.assertEqual(len(ship.un_hit_cells), 1)
        self.assertFalse(ship.is_destroyed)
    
    def test_ship_destroyed(self):
        ship = Ship([(0, 2), (1, 2), (2, 2)])
        ship.hit_ship(0, 2)
        ship.hit_ship(1, 2)
        ship.hit_ship(2, 2)
        
        self.assertEqual(ship.num_hits, 3)
        self.assertEqual(len(ship.un_hit_cells), 0)
        self.assertTrue(ship.is_destroyed)
    
    def test_no_ship_overlap(self):
        ship1 = Ship([(0, 2), (1, 2), (2, 2)])
        ship2 = Ship([(2, 0), (2, 1)])
        
        result = ship1.is_ship_overlap(ship2)
        self.assertFalse(result)
    
    def test_ships_overlap(self):
        ship1 = Ship([(0, 2), (1, 2), (2, 2)])
        ship2 = Ship([(2, 0), (2, 1), (2, 2)])
        
        result = ship1.is_ship_overlap(ship2)
        self.assertTrue(result)


class TestBoard(unittest.TestCase):
    def setUp(self):
        self.board = Board(5)
        
        # Player's POV Board
        # S | S | S | S | S |
        # S | S | - | S | S |
        # - | - | - | S | S |
        # S | - | - | S | S |
        # - | - | - | S | - |
        
        # Enemy's POV Board
        # - | - | - | - | - |
        # - | - | - | - | - |
        # - | - | - | - | - |
        # - | - | - | - | - |
        # - | - | - | - | - |
                
        self.ships_positions = [
            [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
            [(1, 3), (2, 3), (3, 3), (4, 3)],
            [(1, 4), (2, 4), (3, 4)],
            [(1, 0), (1, 1)],
            [(3, 0)]
        ]
        
    def add_ships_to_board(self):
        for position in self.ships_positions:
            self.board.place_ship(*position)
    
    def print_boards(self):
        board = self.board.get_board_for_player()
        print("Player's POV Board")
        Board.print_board(board)
        print("\nEnemy's POV Board")
        board = self.board.get_board_for_enemy()
        Board.print_board(board)

    def test_place_ships(self):
        self.add_ships_to_board()
        self.assertEqual(len(self.board.occupied_cells), 15)
    
    def test_place_ship_on_occupied_cell(self):
        self.board.place_ship(*[(0, 0), (0, 1)])
        with self.assertRaises(BoardException):
            self.board.place_ship(*[(0, 0), (1, 0)])
    
    def test_place_ship_with_invalid_cell(self):
        with self.assertRaises(BoardException):
            self.board.place_ship(*[(6, 0), (5, 0)])
    
    def test_enemy_move_hit(self):
        self.add_ships_to_board()
        
        self.board.enemy_move(0, 0)
        self.board.enemy_move(3, 0)
        
        self.assertEqual(len(self.board.dead_ships), 1)
        self.assertEqual(len(self.board.hit_cells), 2)
        self.assertEqual(len(self.board.un_hit_cells), 13)
        self.assertFalse(self.board.is_player_lost)
        self.assertEqual(len(self.board.valid_moves), 23)
    
    def test_enemy_move_missed(self):
        self.add_ships_to_board()
        
        self.board.enemy_move(1, 2)
        self.board.enemy_move(4, 4)
        
        self.assertEqual(len(self.board.dead_ships), 0)
        self.assertEqual(len(self.board.hit_cells), 0)
        self.assertEqual(len(self.board.un_hit_cells), 15)
        self.assertFalse(self.board.is_player_lost)
        self.assertEqual(len(self.board.valid_moves), 23)
    
    def test_player_lost(self):
        self.board.place_ship((1, 4), (2, 4), (3, 4))
        self.board.place_ship((1, 0), (1, 1))
        
        # print("Before")
        # self.print_boards()
        
        attack_moves = [
            (1, 4), 
            (2, 4), 
            (3, 4), 
            (1, 0), 
            (1, 1)
        ]
        
        for attack_move in attack_moves:
            self.board.enemy_move(*attack_move)
        
        # print("\n\nAfter")
        # self.print_boards()
        
        self.assertEqual(len(self.board.dead_ships), 2)
        self.assertEqual(len(self.board.hit_cells), 5)
        self.assertEqual(len(self.board.un_hit_cells), 0)
        self.assertTrue(self.board.is_player_lost)
        self.assertEqual(len(self.board.valid_moves), 20)


if __name__ == "__main__":
    unittest.main()
