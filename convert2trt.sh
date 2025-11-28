#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

# Script to convert ONNX model to TensorRT engine format

set -e

# Default paths and settings
TRTEXEC_PATH="${TRTEXEC_PATH:-/usr/src/tensorrt/bin/trtexec}"
INPUT_ONNX=""
OUTPUT_ENGINE=""
FP16_MODE=false
INT8_MODE=false
VERBOSE=false

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print usage
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Convert ONNX model to TensorRT engine format.

OPTIONS:
    -i, --input <path>          Input ONNX model file path (required)
    -o, --output <path>         Output TensorRT engine file path (required)
    --trtexec <path>            Path to trtexec binary
                                Default: /usr/src/tensorrt/bin/trtexec
    --fp16                      Enable FP16 mode (default: FP32)
    --int8                      Enable INT8 mode (default: FP32)
    --verbose                   Enable verbose output
    -h, --help                  Display this help message

EXAMPLES:
    # Convert with default settings (FP32)
    $0 -i model.onnx -o model.engine

    # Convert with FP16 mode
    $0 -i model.onnx -o model_fp16.engine --fp16

    # Convert with INT8 mode
    $0 -i model.onnx -o model_int8.engine --int8

EOF
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input)
            INPUT_ONNX="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_ENGINE="$2"
            shift 2
            ;;
        --trtexec)
            TRTEXEC_PATH="$2"
            shift 2
            ;;
        --fp16)
            FP16_MODE=true
            shift
            ;;
        --int8)
            INT8_MODE=true
            shift
            ;;
        --verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            ;;
    esac
done

# Validate inputs
if [[ -z "$INPUT_ONNX" ]] || [[ -z "$OUTPUT_ENGINE" ]]; then
    echo -e "${RED}Error: Input and output paths are required.${NC}"
    usage
fi

if [[ ! -f "$INPUT_ONNX" ]]; then
    echo -e "${RED}Error: Input ONNX file not found: $INPUT_ONNX${NC}"
    exit 1
fi

if [[ ! -f "$TRTEXEC_PATH" ]]; then
    echo -e "${RED}Error: trtexec not found at: $TRTEXEC_PATH${NC}"
    echo -e "${YELLOW}Please install TensorRT or set correct TRTEXEC_PATH environment variable.${NC}"
    exit 1
fi

# Build trtexec command
CMD="$TRTEXEC_PATH --onnx=$INPUT_ONNX --saveEngine=$OUTPUT_ENGINE"

# Add precision mode
if [[ "$FP16_MODE" == true ]]; then
    CMD="$CMD --fp16"
    PRECISION="FP16"
elif [[ "$INT8_MODE" == true ]]; then
    CMD="$CMD --int8"
    PRECISION="INT8"
else
    PRECISION="FP32"
fi

# Add verbose flag if enabled
if [[ "$VERBOSE" == true ]]; then
    CMD="$CMD --verbose"
fi

# Print conversion info
echo -e "${GREEN}======================================${NC}"
echo -e "${GREEN}TensorRT ONNX to Engine Conversion${NC}"
echo -e "${GREEN}======================================${NC}"
echo -e "${YELLOW}Input ONNX:${NC}       $INPUT_ONNX"
echo -e "${YELLOW}Output Engine:${NC}    $OUTPUT_ENGINE"
echo -e "${YELLOW}Precision Mode:${NC}   $PRECISION"
echo -e "${YELLOW}TensorRT Exe:${NC}     $TRTEXEC_PATH"
echo -e "${GREEN}======================================${NC}"
echo ""

# Execute conversion
echo -e "${YELLOW}Starting conversion...${NC}"
echo "Command: $CMD"
echo ""

if eval "$CMD"; then
    echo ""
    echo -e "${GREEN}======================================${NC}"
    echo -e "${GREEN}Conversion completed successfully!${NC}"
    echo -e "${GREEN}======================================${NC}"
    
    # Print file info
    if [[ -f "$OUTPUT_ENGINE" ]]; then
        FILE_SIZE=$(du -h "$OUTPUT_ENGINE" | cut -f1)
        echo -e "${YELLOW}Output Engine Size:${NC} $FILE_SIZE"
    fi
    exit 0
else
    echo ""
    echo -e "${RED}======================================${NC}"
    echo -e "${RED}Conversion failed!${NC}"
    echo -e "${RED}======================================${NC}"
    exit 1
fi
