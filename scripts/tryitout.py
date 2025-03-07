my_list = [1, 2, 3, 4, 5]
your_list = ["a", "b", "c"]

for index, item in enumerate(reversed(my_list)):
    print(index, item)

print(my_list+your_list)

print((my_list)[-3])