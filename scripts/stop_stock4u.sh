#!/bin/bash

echo "========================================"
echo "        Stopping Stock4U Services"
echo "========================================"
echo

echo "Stopping Docker containers..."
docker-compose down

echo
echo "Cleaning up..."
docker system prune -f

echo
echo "Stock4U services have been stopped."
echo
echo "To start again, run: ./start_stock4u.sh"
echo
