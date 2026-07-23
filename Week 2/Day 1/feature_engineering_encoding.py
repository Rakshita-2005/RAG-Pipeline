import pandas as pd

# Customer Purchase Dataset

data = {
    "Customer_ID": [
        101,102,103,104,105,106,107,108,109,110,
        111,112,113,114,115
    ],

    "Age": [
        22,25,31,45,52,36,29,41,55,23,
        34,48,27,39,60
    ],

    "Salary": [
        25000,32000,45000,65000,90000,
        52000,38000,72000,110000,28000,
        60000,85000,35000,67000,120000
    ],

    "Department": [
        "IT","HR","Finance","IT","Marketing",
        "Finance","IT","HR","Marketing","IT",
        "Finance","Marketing","HR","IT","Finance"
    ],

    "Education": [
        "Graduate","Postgraduate","Graduate",
        "PhD","Postgraduate","Graduate",
        "Graduate","Postgraduate","PhD",
        "Graduate","Postgraduate","PhD",
        "Graduate","Postgraduate","PhD"
    ],

    "Experience_Level": [
        "Low","Medium","Medium","High","High",
        "Medium","Low","High","High","Low",
        "Medium","High","Low","Medium","High"
    ],

    "City": [
        "Bangalore","Mumbai","Delhi","Chennai","Hyderabad",
        "Pune","Bangalore","Delhi","Mumbai","Chennai",
        "Hyderabad","Pune","Bangalore","Delhi","Mumbai"
    ],

    "Purchase": [
        0,1,0,1,1,
        1,0,1,1,0,
        1,1,0,1,1
    ]
}


df = pd.DataFrame(data)

print(df)

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())