"""People pipeline: find the athletes, place them on the court and keep track of who is who.

run_people.py runs the stages in order: detect people in each camera (detect.py), combine the
cameras into court positions (fuse_global.py), assemble the fragments into whole people
(assemble.py), sort out who is playing (roles.py), read the shirt numbers (jersey_mv.py),
repair the identities with the numbers (jersey_repair.py) and lock the result to a roster of
ten players (roster_lock.py). Each stage writes a file in work/people/ that the next one
reads, so any of them can be rerun on its own. The 3D bodies are made afterwards by
run_bodies.py.

Each file puts the directory above on sys.path itself, so the scripts run either as scripts
or as modules.
"""
