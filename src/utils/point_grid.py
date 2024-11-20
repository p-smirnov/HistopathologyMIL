

class GridPoint(object):
    def __init__(self, x, y, up_neighbour=None, down_neighbour=None, left_neighbour=None, right_neighbour=None):
        self.x = x
        self.y = y
        self.up_neighbour = up_neighbour
        self.down_neighbour = down_neighbour
        self.left_neighbour = left_neighbour
        self.right_neighbour = right_neighbour
    
    def __str__(self):
        return f"({self.x}, {self.y})"
    
    def __repr__(self):
        return f"({self.x}, {self.y}) with neighbours: {self.up_neighbour}, {self.down_neighbour}, {self.left_neighbour}, {self.right_neighbour}"

    def get_coords(self):
        return (self.x, self.y)

class PointGrid(object):
    def __init__(self, pos_x, pos_y, step_size = 512):
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.points = {}
        self.step_size = step_size
        self._init_points()
        self._initialize_neighbours()
    
    def _init_points(self):
        for i in range(len(self.pos_x)):
            x_pos = self.pos_x[i]
            y_pos = self.pos_y[i]
            self.points[(x_pos, y_pos)] = GridPoint(x_pos, y_pos)

    def _initialize_neighbours(self):
        for point in self.points.values():
            x = point.x
            y = point.y
            point.up_neighbour = self._get_point(x, y + self.step_size)
            point.down_neighbour = self._get_point(x, y - self.step_size)
            point.left_neighbour = self._get_point(x - self.step_size, y)
            point.right_neighbour = self._get_point(x + self.step_size, y)

    def _get_point(self, x, y):
        if (x, y) not in self.points:
            return None
        return self.points[(x, y)]

    def get_square(self, x, y, n_steps = 16):
        square = []
        for i in range(n_steps):
            for j in range(n_steps):
                cur_point = self._get_point(x + i * self.step_size, y + j * self.step_size)
                if cur_point is not None:
                    square.append(cur_point.get_coords())
                else:
                    square.append(None)
        return square
    def get_index(self, x, y):
        index = None
        for i in range(len(self.pos_x)):
            if (self.pos_x[i] == x and self.pos_y[i] == y):
                index = i
                break
        return index