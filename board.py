import copy

class Board:
    W = 1
    B = -1
    EMPTY = 0

    def __init__(self):
        self._game_board = []
        self._last_player = 0
        self._last_move = None
        self._dimension = 0

    def print_board(self):
        pass

    def get_children(self, letter):
        return list()

    def evaluate(self):
        return 0

    def is_terminal(self):
        return True
    
	@property
    def last_move(self):
        return self._last_move
	
    @last_move.setter
    def last_move(self, l):
        self._last_move = l
	
    @property
    def dimension(self):
        return self._dimension
	
    @dimension.setter
    def dimension(self, d):
        self._dimension = d
    
	@property
    def game_board(self):
        return self._game_board
    
	@game_board.setter
    def game_board(self, g):
	    self.game_board = copy.deepcopy(g)
	
    
	@property
    def last_move(self):
        return self._last_move

    @last_move.setter
    def last_move(self, move_obj):
        self._last_move.row = move_obj.row
		self._last_move.col = move_obj.col
		self.last_move.value = move_obj.value