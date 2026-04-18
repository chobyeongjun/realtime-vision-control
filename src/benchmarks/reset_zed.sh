#!/bin/bash
echo "Resetting ZED camera..."
sudo pkill -f "ZED\|zed" 2>/dev/null
sudo systemctl restart zed_x_daemon
sudo systemctl restart nvargus-daemon
sleep 3
echo "ZED ready!"

