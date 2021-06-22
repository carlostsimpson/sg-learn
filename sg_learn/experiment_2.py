"""
this is the end of the main notebook part, further optional cells are included below
also note that the previous basicloop_classificationproof cell can be repeated, cumulating the proof history
"""

# a function to view the initial data with instance input as above
Dd.print_instances()

# InList use case (this isn't covered by the instance chooser)
# modify the following as desired: proving list, training list
proving_instances, training_instances, title_text = Dd.InList(
    [6, 7], [5, 8, 9, 10, 11]
)
HST.reset()
Dd.classificationproof(Mmr, Mm, 0, proving_instances, title_text)
Dd.classificationproof(Mm, Mm, 0, proving_instances, title_text)
Dd.basicloop_classificationproof(
    Mm, Mm, proving_instances, training_instances, title_text
)
