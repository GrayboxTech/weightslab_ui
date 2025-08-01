#!/bin/bash

set -e  # Exit on error

# Load config
source config.env

echo "🔐 Connecting to EC2 at $EC2_HOST..."

ssh -i $KEY_PATH $EC2_USER@$EC2_HOST <<EOF
  set -e
  echo "📦 Installing dependencies and pulling code..."

#   echo "Clone/install weightslab package..."
#   if [ ! -d "$WEIGHTSLAB_DIR" ]; then
#     echo "mkdir -p $WEIGHTSLAB_DIR"
#     mkdir -p $WEIGHTSLAB_DIR
#     echo "git clone $WEIGHTSLAB_REPO $WEIGHTSLAB_DIR"
#     git clone $WEIGHTSLAB_REPO $WEIGHTSLAB_DIR
#     echo "pip install -e $WEIGHTSLAB_DIR"
#     pip install -e $WEIGHTSLAB_DIR
#     cd ..
#   else
#     echo "cd $WEIGHTSLAB_DIR && git pull"
#     cd $WEIGHTSLAB_DIR && git pull
#     echo "pip install -e $WEIGHTSLAB_DIR"
#     pip install -e .
#     cd ..
#   fi

#   # Clone/pull trainer repo and move current directory there
#   if [ ! -d "$TRAINER_DIR" ]; then
#     echo "git clone $TRAINER_REPO $TRAINER_DIR"
#     git clone $TRAINER_REPO $TRAINER_DIR
#     echo "cd $TRAINER_DIR"
#     cd $TRAINER_DIR
#   else
#     echo "cd $TRAINER_DIR && git pull"
#     cd $TRAINER_DIR && git pull
#   fi
#   echo "pip install -r requirements.txt"
#   pip install -r requirements.txt

#   echo "$COMPILE_GRPC"
#   eval $COMPILE_GRPC

  # Launch trainer in tmux
  echo "🚀 Launching trainer in tmux session: $TMUX_SESSION"
  tmux kill-session -t $TMUX_SESSION 2>/dev/null || true
  echo "tmux new -s $TMUX_SESSION -d 'cd $TRAINER_DIR && $TRAIN_CMD > train.log 2>&1'"
  tmux new -s $TMUX_SESSION -d "cd $TRAINER_DIR && $TRAIN_CMD > train.log 2>&1"
EOF

# Setup SSH tunnel
echo "🔁 Setting up local tunnel (port 50051)..."
ssh -i $KEY_PATH -f -N -L 50051:localhost:50051 $EC2_USER@$EC2_HOST

# Start UI
echo "📊 Launching WeightsLab UI locally..."
python3 weights_lab.py --root_directory $LOCAL_DATA_PATH
