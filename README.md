# sg-learn
deep learning about semigroups

This is going to contain a jupyter notebook for classification proofs of nilpotent semigroups with 
a pair of neural networks learning to do the classification proofs minimizing the number of nodes. 
A preliminary version of the associated preprint will also be included in the repository. 

The first part of the notebook should now run with user input to choose the parameters, mainly these
are alpha,beta which are the number of elements of the ``associated-grade'' pieces of the nilpotent
semigroup A, thus alpha is the number of elements of A-A.A  and beta is the number of elements of
A.A - A.A.A.   We are assuming in this implementation that the next associated-graded sizes are 1,1
and 1 is 4-nilpotent so the full associated-graded size vector is (alpha,beta,1,1). 

Use-cases that can be done on cpu include (3,2,1,1), (3,3,1,1), (4,2,1,1) and maybe a little more.
For these, model_n = 4 should be sufficient. 
On GPU I've done up to (5,3,1,1) and (4,5,1,1), for these use model_n = 6 or 8. 



