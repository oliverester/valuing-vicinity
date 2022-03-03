python -m ship_ai.pre_histo \
--wsi_path /projects/praediknika/data/from_mhh/RCCs_HE-El_svs \
--patch_size 256 \
--patch_overlap 0 \
--downsample 8 \
--max_background_ratio 0.98 \
--annotation_paths /homes/oester/repositories/prae/data/segmentation/annotations_heel_seg \
--annotation_extension json \
--exclude_classes tumor \
--tissue_annotation tissue \
--intersection_ratio 0.1 \
--tissue_annotation tissue \
sample_patches \
--label_map_file /homes/oester/repositories/prae/data/segmentation/label_map.json \
--db_path /homes/oester/repositories/prae/data/segmentation/preprocessed/HEEL_seg_masks   \
--storage_type disk \
--incomplete_annotations \
--check_resolution 0.2519 \
--annotation_output intersection_mask \
--patches_per_batch 100 \
--normalize_stains \
--normalize_file /homes/oester/repositories/pre-histo/scripts/prae/stain_vectors/RCC-TA-101.001~C_vector_normalization.json \
--overwrite

