import streamlit as st
import numpy as np
import face_recognition
import pandas as pd
import cv2
from io import BytesIO
import os
from datetime import datetime

# Define file names
data_file = "registered_faces.xlsx"
attendance_file = "attendance.xlsx"

# Ensure previous attendance file is cleared once
if not os.path.exists(attendance_file):
    pd.DataFrame(columns=["Name", "Registration Number", "Attendance Status", "Subject", "Year", "Branch & Section", "Date", "Hour"]).to_excel(attendance_file, index=False)

# Load Registered Users Data
try:
    df = pd.read_excel(data_file)
    df["Encoding"] = df["Encoding"].apply(lambda x: np.fromstring(x[1:-1], sep=','))
except FileNotFoundError:
    st.error("No registered users found! Please register first.")
    df = pd.DataFrame(columns=["Name", "Registration Number", "Branch", "Section", "Year", "Encoding"])

# Function to recognize faces with stricter accuracy
def recognize_faces(image):
    image_bytes = BytesIO(image.read())
    frame = np.frombuffer(image_bytes.getvalue(), dtype=np.uint8)

    if frame.size == 0:
        st.error("Uploaded image is empty or corrupted.")
        return []
    
    frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)
    if frame is None:
        st.error("Failed to decode image. Please upload a valid image.")
        return []
    
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_encodings = face_recognition.face_encodings(rgb_frame)

    if not face_encodings:
        st.warning("No faces detected in the uploaded image.")
        return []

    present_users = set()
    known_encodings = df["Encoding"].tolist()
    
    for encoding in face_encodings:
        distances = face_recognition.face_distance(known_encodings, encoding)
        best_match_index = np.argmin(distances)

        if distances[best_match_index] < 0.5:
            present_users.add(df.iloc[best_match_index]["Name"])

    return list(present_users)
# Function to mark attendance
def mark_attendance(present_users, date, subject, hour, branch, section, year):
    date_str = date.strftime("%Y-%m-%d")

    # Filter registered students based on selection
    filtered_df = df[(df["Branch"] == branch) & 
                     (df["Section"] == section) & 
                     (df["Year"] == year)]
    
    if filtered_df.empty:
        st.error("No registered students found for the selected criteria.")
        return pd.DataFrame()

    # Create an attendance DataFrame
    attendance = pd.DataFrame({
        "Name": filtered_df["Name"],
        "Registration Number": filtered_df["Registration Number"],
        "Attendance Status": ["Present" if name in present_users else "Absent" for name in filtered_df["Name"]],
        "Subject": subject,
        "Year": year,
        "Branch & Section": branch + " - " + section,
        "Date": date_str,
        "Hour": hour
    })

    # Load existing attendance data
    try:
        existing_attendance = pd.read_excel(attendance_file)
    except FileNotFoundError:
        existing_attendance = pd.DataFrame()
    
    # Append new attendance data
    updated_attendance = pd.concat([existing_attendance, attendance], ignore_index=True)
    updated_attendance.to_excel(attendance_file, index=False)
    
    return attendance


# Function to view attendance with filtering
def view_attendance():
    st.subheader("View Attendance Records")
    search_by = st.radio("Search By", ["Registration Number", "Name"])
    search_value = st.text_input(f"Enter {search_by}")
    
    month_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    from_month = st.selectbox("From Month", month_names)
    to_month = st.selectbox("To Month", month_names)
    
    if st.button("Get Attendance Records"):
        try:
            attendance_df = pd.read_excel(attendance_file)
            attendance_df['Date'] = pd.to_datetime(attendance_df['Date'])
            
            from_month_num = month_names.index(from_month) + 1
            to_month_num = month_names.index(to_month) + 1
            filtered_df = attendance_df[(attendance_df['Date'].dt.month >= from_month_num) & (attendance_df['Date'].dt.month <= to_month_num)]
            
            if search_by == "Registration Number":
                filtered_df = filtered_df[filtered_df["Registration Number"].astype(str) == search_value]
            else:
                filtered_df = filtered_df[filtered_df["Name"].str.contains(search_value, case=False, na=False)]
            
            if filtered_df.empty:
                st.warning("No records found for the given criteria.")
            else:
                st.dataframe(filtered_df)
                output_file = "filtered_attendance.xlsx"
                filtered_df.to_excel(output_file, index=False)
                st.download_button(label="Download Attendance Records", data=open(output_file, "rb").read(), file_name=output_file, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        except FileNotFoundError:
            st.error("No attendance records found!")
# Function to calculate attendance percentage per subject
def calculate_attendance_percentage(month):
    try:
        attendance_df = pd.read_excel(attendance_file)
        attendance_df['Date'] = pd.to_datetime(attendance_df['Date'])
        monthly_attendance = attendance_df[attendance_df['Date'].dt.month == month]
        
        if monthly_attendance.empty:
            st.error(f"No attendance records found for {month}.")
            return

        subject_summary = monthly_attendance.groupby(["Registration Number", "Name", "Subject"]).agg(
            Classes_Conducted=("Attendance Status", "size"),
            Classes_Attended=("Attendance Status", lambda x: (x == "Present").sum())
        ).reset_index()

        pivot_table = subject_summary.pivot(index=["Registration Number", "Name"], columns="Subject", values=["Classes_Conducted", "Classes_Attended"]).fillna(0)
        pivot_table.columns = pd.MultiIndex.from_tuples([(col[1], col[0]) for col in pivot_table.columns])
        pivot_table.reset_index(inplace=True)
        
        pivot_table[('Total', 'Classes_Conducted')] = pivot_table.filter(like="Classes_Conducted").sum(axis=1)
        pivot_table[('Total', 'Classes_Attended')] = pivot_table.filter(like="Classes_Attended").sum(axis=1)
        
        pivot_table[('Total', 'Percentage')] = pivot_table[('Total', 'Classes_Attended')].div(pivot_table[('Total', 'Classes_Conducted')]).mul(100).fillna(0)
        
        st.subheader("Attendance Summary by Subject")
        st.dataframe(pivot_table)
    
    except FileNotFoundError:
        st.error("No attendance records found!")

# Streamlit UI
def main():
    st.title("Face Recognition Attendance System")
    menu = ["Mark Attendance", "View Attendance", "Total Attendance Percentage"]
    choice = st.sidebar.selectbox("Menu", menu)
    
    if choice == "Mark Attendance":
        st.subheader("Upload Image to Mark Attendance")
        date = st.date_input("Select Date")
        subject = st.selectbox("Select Subject", ["ECA", "ADC", "LCS", "BE", "EMTL", "SS", "ADC(lab)", "ECA(lab)", "DTI"])
        hour = st.selectbox("Select Hour", [1, 2, 3, 4, 5, 6])
        branch = st.selectbox("Select Branch", ["ECE", "CSE", "CSE DS", "CSE AIML"])
        section = st.selectbox("Select Section", ["A", "B", "C", "D", "E"])
        year = st.selectbox("Select Year", ["1st Year", "2nd Year", "3rd Year", "4th Year"])
        
        image = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])
        
        if image and date and subject and branch and section and year:
            present_users = recognize_faces(image)
            if present_users:
                attendance = mark_attendance(present_users, date, subject, hour, branch, section, year)
                if not attendance.empty:
                    st.success("Attendance Updated Successfully!")
                    st.dataframe(attendance)
    
    elif choice == "View Attendance":
        view_attendance()
    
    elif choice == "Total Attendance Percentage":
        st.subheader("View Total Attendance Percentage")
        month_names = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
        month = st.selectbox("Select Month", month_names)
        month_number = month_names.index(month) + 1
        if st.button("Calculate"):
            calculate_attendance_percentage(month_number)

if __name__ == "__main__":
    main()
