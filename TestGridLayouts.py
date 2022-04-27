# Simple grid map; start position is in bottom left corner, goal is in top right
    sx = 1  # [m]
    sy = 1  # [m]
    gx = [49]  # [m]
    gy = [49]  # [m]

# Simple grid map; start position is in bottom left corner, goal is in bottom right
    sx = 1 # [m]
    sy = 1 # [m]
    gx = [48, 48, 49, 49] # [m]
    gy = [1, 2, 1, 2] # [m]

# Simple grid map; start position is in bottom left corner, goal is in top left
    sx = 1 # [m]
    sy = 1 # [m]
    gx = [1, 2, 1, 2] # [m]
    gy = [48, 48, 49, 49] # [m]

# Simple obstacle map
    sx = 1  # [m]
    sy = 1  # [m]
    gx = [49]  # [m]
    gy = [49]  # [m]
    square(10, 20, 10, 20)
    square(10, 20, 30, 40)
    square(40, 50, 20, 30)
    square(30, 40, 40 , 50)

# Complex wind-field map
    sx = 48  # [m]
    sy = 20  # [m]
    gx = [8]  # [m]
    gy = [44]  # [m]
    square(10, 15, 10, 15)
    square(10, 14, 30, 35)
    square(40, 45, 20, 25)
    square(30, 35, 40 , 45)
    square(24, 30, 0, 9)
    square(18, 21, 33, 45)
    square(44, 50, 3, 6)
    square(27, 30, 24 , 29)

# Realistic variable-wind-field map
        sx = 4  # [m]
        sy = 25  # [m]
        gx = [46]  # [m]
        gy = [23]  # [m]
        square(6, 10, 5, 12)

        square(8, 9, 19, 44)
        square(14, 15, 19, 44)
        square(10, 13, 19, 20)
        square(10, 13, 43, 44)
        square(14, 30, 38, 39)
        square(14, 30, 43, 44)
        square(29, 30, 38, 44)

        square(13, 26, 5, 10)
        square(22, 26, 10, 22)
        square(18, 18, 11, 14)
        square(19, 21, 14, 14)

        square(21, 28, 27, 27)
        square(22, 28, 28, 28)
        square(23, 28, 29, 29)
        square(24, 28, 30, 30)
        square(25, 28, 31, 31)
        square(26, 28, 32, 32)
        square(27, 28, 33, 33)
        square(28, 28, 34, 34)

        square(38, 42, 5, 32)
        square(30, 37, 8, 11)

        square(33, 34, 38, 44)

        square(38, 42, 38, 44)
