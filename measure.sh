#!/bin/bash

if [ $# -eq 0 ]; then
  echo "Please provide the Python script file as an argument."
  exit 1
fi

python_script="$1"
current_datetime=$(date +"%Y%m%d%H%M%S")
result_file="results/$(basename "$python_script" .py)_$current_datetime.txt"

# Function to capture CPU and memory utilization
get_cpu_mem_util() {
  local pid="$1"
  cpu_util=$(ps -p "$pid" -o %cpu,cmd | awk '{print $1}')
  mem_util=$(ps -p "$pid" -o %mem,cmd | awk '{print $1}')
  echo -e ",$cpu_util, $mem_util"
}

# Flag to indicate if headers have been printed
headers_printed=false

# Start the Python script
python3 "$python_script" &
pid=$!

while true; do
  # Check if process has finished
  if ! kill -0 "$pid" 2> /dev/null; then
    echo "Process $pid has finished running."
    break
  fi

  # Capture CPU and memory utilization
  data=$(get_cpu_mem_util "$pid")

  # Print headers only once
  if ! "$headers_printed"; then
    echo "CPU Utilization,Memory Utilization" >> "$result_file"
    headers_printed=true
  fi

  # Write data to CSV file
  echo "$data" >> "$result_file"

  # Sleep for 1 second
  sleep 1
done
