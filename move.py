class Move:
    def __init__(self, row, col, value):
        self._row = row
        self._col = col
        self._value = value
    
	@property
    def row(self):
        return self._row
    
	@property
    def col(self):
        return self._col
    
	@property
    def value(self):
        return self._value
    
	@row.setter
    def row(self, row):
        self._row = row
    
	@col.setter
    def col(self, col):
        self._col = col
    
	@value.setter
    def value(self, value):
        self._value = value