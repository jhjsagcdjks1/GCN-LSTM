<font size="5"><strong>A GCN-LSTM Based on Adaptive Number of Layers for Reconstructing Temporal Gene Regulatory Networks</strong></font>
![image][(https://github.com/jhjsagcdjks1/GCN-LSTM/blob/main/Framework.png)]
***
**Dependencies**
• 'networkx==2.2'；  
• 'numpy'；  
• 'scikit-learn'；  
• 'scipy'；  
• 'tensorflow==1*'
***
**Usage**
Preparing for gene expression profiles and gene-gene adjacent matrix

The GCN-LSTM model integrates gene expression matrices (N×M) across n time points with prior gene topology (N×N) to learn low-dimensional vectorized representations at each time point under supervised learning conditions, and to generate gene features for the next time point. By converting the prior gene topology into the `.npz` format as shown in the example and running it in the GCN-LSTM model, the gene regulatory network for the next time point can be obtained.

Command to run GCN-LSTM
 python main.py
