#!/bin/bash

echo "Adding CRAN repository for R..."
sh -c 'echo "deb https://cloud.r-project.org/bin/linux/ubuntu jammy-cran40/" > /etc/apt/sources.list.d/r-project.list'

echo "Adding GPG key"
apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E298A3A825C0D65DFD57CBB651716619E084DAB9

echo "Updating package list..."
apt-get update

echo "Installing R..."
apt-get install -y r-base
apt-get install -y r-base-dev
echo "R installation completed."

echo "Installing R packages"
Rscript ./docker/install_r_packages.R
echo "R packages installed"
