p1 = (0, 6)
vel = (4, 4)
p2 = (4, 10)

points = []

x = p1[0]
for y in range(p1[1], p2[1] + 1):

    points.append((x, y))
    if x != p2[0]:
        x += 1

print(points)
