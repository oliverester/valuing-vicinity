# Project PraeDikNika

The project _PraeDikNika_ is a cooperative research project at the UKE (Universitätsklinikum Essen) and the MHH (Medizinische Hochschule Hannover).
Research Question:
- Can we develop a Deep Learning algorithm to predict the treatment response given a cancer immuno-oncology therapy for renal cell carcinoma (RCC) based on histopathologic imaging?
- Can we develop a Deep Learning segmentation algorithm to detect multiple cell structures of a RCC WSI?


# Data 

The data comprises 28 fully annotated WSIs (15 bad-responder, 13 good-responder) in svs-format. 
There are 6 different stainings:

- CD4+FoxP3
- CD8+CD20
- CD68
- CD204
- HEEL
- PDL1

Example WSI with annotations:

<img src="https://git.uni-due.de/tio/projects/prehisto/uploads/6fcbac9a7c605f8315ed81be877071ca/Bildschirmfoto_2021-11-17_um_08.44.18.png"  width="50%" height="50%">

The data is preprocessed via the [prehisto-libraray ](https://git.uni-due.de/tio/projects/prehisto) into patches. The patches are normalized via the Mancenko method.

Path: NFS:/projects/praediknika/data


# Project members

- Prof. Dr. med. Jan Hinrich Bräsen (MHH, Facharzt für Pathologie und Innere Medizin, Oberarzt)
- Prof. Dr. Dr. Jens Kleesiek (UKE)
- Prof. Dr. Viktor Grünwald (UKE & MHH)
- Oliver Ester (UKE, Research Employee)
- Jessica Schmitz (MHH, Wissenschaftliche Leitung)
- et al.

# Timeline

| Phase | Time |
| ------ | ------ |
| Data Transfer | 1.10.2021 |
| Data Preprocessing | - 20.10.2021 |
| Patch-based Treatment Outcome | - 16.11.2021 |
| WSI-based Treatment Outcome | - ? |
| Segmentation algorithm | - ? |
