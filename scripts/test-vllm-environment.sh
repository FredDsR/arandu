#!/bin/bash

# Test vLLM deployment prerequisites script
# Clear output formatting and color codes
RED="\e[31m"
GREEN="\e[32m"
YELLOW="\e[33m"
RESET="\e[0m"

# Display instructions
echo -e "${YELLOW}Running vLLM Deployment Prerequisites Test${RESET}"

echo "Please ensure you are running this script on PCAD tupi nodes."

echo -e "\n${GREEN}Checking Ray installation...${RESET}"
if ! command -v ray &> /dev/null; then
    echo -e "${RED}Ray is NOT installed.${RESET}"
else
    echo -e "${GREEN}Ray is installed.${RESET}"
fi


echo -e "\n${GREEN}Checking NCCL installation...${RESET}"
if ! command -v nccl &> /dev/null; then
    echo -e "${RED}NCCL is NOT installed.${RESET}"
else
    echo -e "${GREEN}NCCL is installed.${RESET}"
fi


echo -e "\n${GREEN}Checking CUDA installation...${RESET}"
if ! command -v nvcc &> /dev/null; then
    echo -e "${RED}CUDA is NOT installed.${RESET}"
else
    echo -e "${GREEN}CUDA is installed.${RESET}"
fi


echo -e "\n${GREEN}Checking SLURM installation...${RESET}"
if ! command -v srun &> /dev/null; then
    echo -e "${RED}SLURM is NOT installed.${RESET}"
else
    echo -e "${GREEN}SLURM is installed.${RESET}"
fi


echo -e "\n${GREEN}Checking network connectivity...${RESET}"
ping -c 4 google.com
if [ $? -eq 0 ]; then
    echo -e "${GREEN}Network is accessible.${RESET}"
else
    echo -e "${RED}Network is NOT accessible.${RESET}"
fi


echo -e "\n${GREEN}Checking HuggingFace availability...${RESET}"
if curl --head --silent --fail https://huggingface.co; then
    echo -e "${GREEN}HuggingFace is available.${RESET}"
else
    echo -e "${RED}HuggingFace is NOT available.${RESET}"
fi


# Make the script executable
chmod +x scripts/test-vllm-environment.sh

echo -e "\n${GREEN}Test complete. Please check the output above for any issues.${RESET}"