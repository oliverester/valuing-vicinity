#!/bin/bash
# either poetry run python -m ship_ai.pre_histo.save_template_vectors or poetry run save-template
python -m ship_ai.pre_histo \
        --wsi_path /projects/praediknika/data/wsis/HEEL/bad/RCC-TA-101.001~C.svs \
        --patch_size 256 \
        --patch_overlap 0 \
        --downsample 4 \
        --annotation_paths /homes/oester/repositories/prae/data/annotations_corrected \
        --tissue_annotation tissue \
        --intersection_ratio 0.95 \
        save_template_vectors \
        --vector_store_path scripts/prae/stain_vectors
