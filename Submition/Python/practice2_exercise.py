count = 0
for number in range(1,10):
    if number % 2== 0:
        count += 1
        print(number)
print(f"We have {count} even numbers")

def greet( first_name, last_name):
    print(f"Hi {first_name} {last_name}, Welcome to my code")
    print("hihi")

greet("Anh", "Bui")

#function use to perform a task or to return a value

def get_gretting(name):
    return f"Hi {name}"

message = get_gretting("Anh")
print(message)

def increment(number, by):
    return number + by
result = increment(2,1)

print(increment(number=4,by=1))

def multiply(*numbers):
    total = 1
    for number in numbers:
        total *= number
    return total

print(multiply(4, 5, 6, 7))

job_skills = ["communication", "excel", "SQL"]
job_skills.remove("excel") # remove a string to the list
print(job_skills)
job_skills.append("excel") # add a string to the list
print(job_skills)

print(len(job_skills))

job_skills.insert(2, "tableau") #insert a string into the list at the index 2
print(job_skills)

job_skills.pop(2) #remove the string at the index number 2
print(job_skills)

#SLICING
job_skills[0:1]
print(job_skills)
job_skills[:]
print(job_skills)

print(job_skills)

print(job_skills)

import os

file_path = r"C:\Users\hoang\Documents\Summer Research\python (locally download file)\used_tesla_tweets_extracted_first_version.csv"

if not os.path.exists(file_path):
    print("❌ File not found. It may still be in the cloud (OneDrive).")
else:
    print("✅ File is ready to use.")