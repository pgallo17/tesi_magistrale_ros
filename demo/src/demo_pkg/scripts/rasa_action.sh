#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

BOT_DIR="${SCRIPT_DIR}/../../../../fiera_bot"

cd $BOT_DIR

rasa run actions