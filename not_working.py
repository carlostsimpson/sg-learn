
#######################################################################################

##### this is the end of the main notebook part, further optional cells are included below
##### also note that the previous basicloop_classificationproof cell can be repeated, cumulating the proof history
def stopcompile():

#######################################################################################

##### a function to view the initial data with instance input as above
Dd.print_instances()

###### InList use case (this isn't covered by the instance chooser)
###### 
## modify the following as desired: proving list, training list 
proving_instances, training_instances, title_text = Dd.InList([6,7],[5,8,9,10,11])  
#
HST.reset()
Dd.classificationproof(Mmr,Mm,0,proving_instances,title_text)
Dd.classificationproof(Mm,Mm,0,proving_instances,title_text)

Dd.basicloop_classificationproof(Mm,Mm,proving_instances,training_instances,title_text)

#######################################################################################

def stopcompile():

#######################################################################################

#######################################################################################

#### do the proofs in a range

for i in range(13):
    print("instance",i)
    proving_instances, training_instances, title_text = Dd.InOne(i)
    Dd.classificationproof(Mm,Mm,0,proving_instances,title_text)

#######################################################################################

def stopcompile():

#######################################################################################


    
        


Pp.profile_filter_on = True
Pp.halfones_filter_on = True

MH = MinimizerHistory(Mmr)  # asks to choose sigma
MH.minimize_all()  # this does all the cases in a row

def stopcompile():

Min = Minimizer(Mm,3,0,0,0)  # individual cases: sigma, x, y, p
Min.check_done_print()
#time.sleep(60)

Min = Minimizer(Mm,3,0,0,1) # individual cases: sigma, x, y, p
Min.check_done_print()
#time.sleep(60)

Min = Minimizer(Mm,3,0,0,2) # individual cases: sigma, x, y, p 
Min.check_done_print()
#time.sleep(60)




Fws = FindWeirdStuff(Dd,Mm)

### a few things to do with that

Fws.tracer_root(5)

Fws.tracer_subroot(5,0,0,0)

Fws.show_cut_column(5,2,1)

Fws.print_one_sample_from_box(0.25,0.35,0.0,0.2)

Fws.print_one_sample_from_box(0.0,0.05,0.0,0.2)
