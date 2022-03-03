python -m ship_ai.pre_histo \
--wsi_path /projects/praediknika/data/wsis/CD4+FoxP3 \
--patch_size 256 \
--patch_overlap 0 \
--downsample 4 \
--max_background_ratio 0.98 \
--annotation_paths /homes/oester/repositories/prae/data/annotations_corrected \
--annotation_extension json \
--intersection_ratio 0.8 \
--tissue_annotation tissue \
sample_patches_and_plot \
--label_map_file ship_ai/examples/label_map.json \
--db_path /homes/oester/repositories/prae/data/preprocessed/CD4+FoxP3   \
--plot_path plot \
--storage_type disk \
--check_resolution 0.2519 \
--patches_per_batch 100 \
--normalize_stains \
--normalize_file /homes/oester/repositories/pre-histo/scripts/prae/stain_vectors/RCC-TA-101.042~B_vector_normalization.json \
--overwrite
