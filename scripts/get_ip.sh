#!/bin/bash

ip addr show tun0 | grep -oP 'inet \K[\d.]+'
