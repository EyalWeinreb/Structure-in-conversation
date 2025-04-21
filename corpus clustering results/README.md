This folder contains the results of the main clustering runs performed in our paper. 
A file is supplied for each corpus analyzed.

The corpuses analyzed were:

*   `CH`: The CallHome corpus
*   `SBC`: The Santa Barbara corpus
*   `CT_B3`: "Edge of Eternity" by Ken Follett, read by John Lee
*   `SOIAF_R1_B5`: "A Dance with Dragons" by George R. R. Martin, read by Roy Dotrice 
*   `SOIAF_R3_B5`: "A Dance with Dragons" by George R. R. Martin, read by @DavidReadsASoIaF


The files supplied are .csv format text files compressed in .zip format. They contain a row for each IU in the corpus and the following columns:

*   `media`: The audio file from which the IU was taken
*   `is_idn`: IU ID number
*   `iu_t_min`: The IU start time in seconds
*   `iu_t_max`: The IU end time in seconds
*   `viu_t_len`: The IU vector length in seconds 
*   `good`: Indicator if the IU was considered good enough to attempt it's clustering
*   `cluster`: The final assigned cluster ID
*   `cluster_n`: Number of IUs in the assigned cluster
*   `cluster_population`: Total number of IUs in the cluster's originating act
*   `cluster_ratio`: `cluster_n / cluster_population`
*   `act`: The length-based clustering subset the IU belongs to (e.g., `decile_01`, `len_850ms-1230ms`).
*   `hunk`: The specific processing group (same as `act` unless the cluster originates from a "leftovers" reclustering attempt, e.g., `decile_01_leftovers_1`).

In addition, each .zip file contains a text file describing how the audio files referenced in the 'media' column were generated from the original source audio files. 

