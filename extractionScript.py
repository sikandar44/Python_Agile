import re
import os

# Define the regular expression patterns to match the log messages
input_pattern = r'(\d{4}/\w+/\d{2}\s\d{2}:\d{2}:\d{2}\.\d+)\s(\w+)\sPORTEVENT\s(\w+\.ttcn):\d+\(function:(\w+)\)\sReceive\soperation\son\sport\s(\w+)\[(\d+)\]\ssucceeded,\smessage\sfrom\s(\d+):\s@variables\.(\w+)\s:\s{([\s\S]*?)\}\sid\s(\d+)'
output_pattern = r'(\d{4}/\w+/\d{2}\s\d{2}:\d{2}:\d{2}\.\d+)\s(\d+)\sPORTEVENT\s(\w+\.ttcn):\d+\(function:(\w+)\)\sSent\son\ssipInternalPort\sto\s(\w+)\s@variables\.(\w+)\s:\s{([\s\S]*?)\}'

# Set the path to the folder containing the log files
log_folder = r"C:\Users\Huseyn Mirzayev\Desktop\Log Analysys data"

# Initialize a counter for the processed txt files
num_processed_files = 0

# Iterate over the log files in the folder
for filename in os.listdir(log_folder):
    if filename.endswith('.txt'):
        # Increment the counter for processed files
        num_processed_files += 1

        # Open the log file and read its contents with utf-8 encoding
        with open(os.path.join(log_folder, filename), 'r', encoding='utf-8') as f:
            log_file = f.read()

        # Check if the log file is empty
        if len(log_file.strip()) == 0:
            print(f"The log file {filename} is empty.")
            continue

        # Use the re.findall() function to extract all matching input and output log messages
        input_messages = re.findall(input_pattern, log_file)
        output_messages = re.findall(output_pattern, log_file)

        # If there are no matching messages, print a message indicating the file is empty
        if not input_messages and not output_messages:
            print(f"The log file {filename} has no input_messages and output_messages.")
        else:
            # Format and write the matching input messages to a file
            input_file_name = os.path.splitext(filename)[0] + '_input.txt'
            with open(os.path.join(log_folder, input_file_name), 'w') as input_file:
                for message in input_messages:
                    date_time, component_id, component_name, function_name, recipient, index, sender, message_type, message_body, msg_id = message
                    formatted_message = f"at {date_time} the component {component_id} received a message of type @{message_type} to {recipient} with the body {message_body}"
                    input_file.write(formatted_message + '\n')

            # Format and write the matching output messages to a file
            output_file_name = os.path.splitext(filename)[0] + '_output.txt'
            with open(os.path.join(log_folder, output_file_name), 'w') as output_file:
                for message in output_messages:
                    date_time, component_id, component_name, function_name, recipient, message_type, message_body = message
                    formatted_message = f"at {date_time} the component {component_id} sent a message of type @{message_type} to {recipient} with the body {message_body}"
                    output_file.write(formatted_message + '\n')

# Print the total number of processed txt files
print(f"{num_processed_files} txt files have been processed.")

