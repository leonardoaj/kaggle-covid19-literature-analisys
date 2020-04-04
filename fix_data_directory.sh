#!/bin/bash

for directory in biorxiv_medrxiv comm_use_subset custom_license noncomm_use_subset; do
 mv data/$directory/$directory/* data/$directory
 rm -rf data/$directory/$directory
 mv data/$directory/pdf_json/* data/$directory
 rm -rf data/$directory/pdf_json
 mv data/$directory/pmc_json/* data/$directory
 rm -rf data/$directory/pmc_json
done
