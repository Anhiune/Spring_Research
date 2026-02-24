student_count = 1000
rating = 4.99  # float
is_published = False
course_name = "Python Programing"  # string
print(student_count)
print(len(course_name))
print(course_name[0])
print(course_name[-1])
print(course_name[0:3])
print(course_name[:3])
course = "Python Programing"
# \" = escape " character, it means make the " visible in the text
# \' = escape ' character, it means make the ' visible in the text
# \\ = escape \ character, it means make the \ visible in the text
# \n = new line
print(course)

first = "Mosh"
last = "Hamedni"
full = f"{len(first)} {2 + 2}"
print(full)

print(course.upper())
print(course.lower())
print(course.title())
print(course.strip())
print(course.find("pro"))
print("pro" in course)
print("Pro" in course)

print(10 + 3)
print(10-3)
print(10 * 3)
print(10/3)
print(10//3)
print(10%3)
print(10**3)

print(round(2.9))
print(abs(-2.7))

x = input("x: ")
y = int(x) + 1
print(f"x= {x}, y: {y}")

successful = False
for number in range(3):
    print("Attempt")
    if successful:
        print("Successful")
        break
else:print("Attemted 3 times and failed")


for x in range(7):
    for y in range(3):
        print(f"x: {x}, y: {y}")


print(type(5))
print(type(range(5)))
# iterable
for x in "Python":
    print(x)

for item in shopping_cart:
    print(item)

number = 100
while number > 0:
    print(number)
    number //=2

command = ""
while command.lower != "quit":
    command = input(">")
    print("ECHO", command)

while True:
    command = input(">")
    print("ECHO", command)
    if command.lower == "quit":
        break

max = 10
while x < max:
    if x // 2 != 0:
        print("x")
    else:
        break



