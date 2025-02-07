# Standardizes a variety of time string formats to HH:MM format.
 
import re

def standardize_time(time_str):
    time_str = time_str.lower().strip()  # Handle case and leading/trailing spaces

    # Count digits; if more than 4, it is an invalid input
    if len(re.findall(r"\d", time_str)) > 4:
      return None
    
    # Case 1: HH:MM AM/PM format (e.g., 6:00 pm, 5:30 am)
    match = re.match(r"(\d{1,2}):?(\d{2})\s*(am|pm)", time_str)
    if match:
        hour = int(match.group(1))
        minute = int(match.group(2))
        ampm = match.group(3)
        if ampm == "pm" and hour != 12:
            hour += 12
        elif ampm == "am" and hour == 12:
          hour = 0
        if 0 <= hour <=23 and 0<= minute <=59:
          return f"{hour:02}:{minute:02}"

    # Case 2: HH AM/PM format (e.g., 6 pm, 8 am, 12 pm)
    match = re.match(r"(\d{1,2})\s*(am|pm)", time_str)
    if match:
        hour = int(match.group(1))
        ampm = match.group(2)
        if ampm == "pm" and hour != 12:
            hour += 12
        elif ampm == "am" and hour == 12:
          hour = 0
        if 0 <= hour <=23:
          return f"{hour:02}:00"
    
    # Case 3: HH:MM format (e.g., 02:00)
    match = re.match(r"(\d{1,2}):(\d{2})", time_str)
    if match:
      hour = int(match.group(1))
      minute = int(match.group(2))
      if 0 <= hour <=23 and 0<= minute <=59:
        return f"{hour:02}:{minute:02}"
    
    # Case 4: HH: (e.g., 18:)
    match = re.match(r"(\d{1,2}):\s*", time_str)
    if match:
      hour = int(match.group(1))
      if 0 <= hour <=23:
          return f"{hour:02}:00"
    
    # Case 5: Single HH format(e.g., 18)
    match = re.match(r"(\d{1,2})$", time_str)
    if match:
      hour = int(match.group(1))
      if 0 <= hour <=23:
          return f"{hour:02}:00"

    # Case 6: 24 Hour Format  (e.g., 1800, 2030)
    match = re.match(r"(\d{2})(\d{2})", time_str)
    if match:
      hour = int(match.group(1))
      minute = int(match.group(2))
      if 0 <= hour <=23 and 0<= minute <=59:
          return f"{hour:02}:{minute:02}"
    
    # Case 7: Single digits followed by 00
    match = re.match(r"(\d{1,2})00", time_str)
    if match:
      hour = int(match.group(1))
      if 0 <= hour <=23:
          return f"{hour:02}:00"

    return None  # Return None for unparsable formats


def main():
  
  # Main function to test the standardize_time function
  
  # Test cases
  test_cases = [
    "600 pm",
    "530 am",
    "1800",
    "6 pm",
    "8 am",
    "12 pm",
    "800",
    "1200",
    "1800",
    "2030",
    "1:30 pm",
    "1:30am",
    "1:00 am",
    "02:00",
    "12:00pm",
    "12:00am",
    "0000",
    "0100",
    "2300",
    "18:",
    "18"
  ]

  fail_cases = [
      "25:00",
      "12345",
      "abc",
      "2400",
      "12:60",
      "13 pm",
      "1234567",
      "12:300",
      "1:234"
    ]
  
  print("---Test Cases---")
  for time_str in test_cases:
      standardized_time = standardize_time(time_str)
      print(f"Original: '{time_str}', Standardized: '{standardized_time}'")
      
  print("\n---Fail Cases---")
  for time_str in fail_cases:
      standardized_time = standardize_time(time_str)
      print(f"Original: '{time_str}', Standardized: '{standardized_time}'")

if __name__ == "__main__":
    main()