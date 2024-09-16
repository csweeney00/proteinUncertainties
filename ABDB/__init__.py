'''
ABDB - Antibody Database python API

This is the backend to U{SAbDab<http://opig.stats.ox.ac.uk>}.
It can also be used to interface with a local copy of the database.

Requirements:

    o A local copy of the data files.

    o Python 2.6 or higher
        o numpy
        o scipy
        o matplotlib
        o biopython (v 1.6+)

    o ANARCI numbering software (now as a separate package)
        
    Proprietry:
        o abysis (abnum)
            or
        o in-house numbering software 
        
    Network access:
        o pdb ftp: ftp://ftp.wwpdb.org/pub/pdb  ( or access to internal pdb mirror )
        o ligand expo site: http://ligand-expo.rcsb.org/files/    (three letter HETATM resid code will be sometimes be sent out)
        o IMGT website: http://www.imgt.org/ ( data retrieved on update )
        
        Optional:
            o abnum (only if no numbering software available) http://www.bioinf.org.uk/abs/abnum/  (sequence data will be sent out ) 
            
    Optional:
        required for updating database:
            o muscle
        required for non-redundant set creation:
            o cd-hit

            
Contents:

    o The structures of all antibody structures in the protein data bank (PDB)
    o Annotations to all of these structures. 
    o API to access the database and retrieve information about structures.

    o Tools for analysis (not fully released)


Example:

For more information see the documentation for ABDB.Database_interface

@author: James Dunbar
@contact: james.dunbar-[at]-dtc.ox.ac.uk

'''

database_path = ""
muscle_path = ""
numbering_software_path = None
derived_mirror_path = None
structure_mirror_path = None

allow_online = False
numbering_software = True
abysis = False
anarci_available = True
muscle = True
allow_updating = True
allow_ftp = True
use_mirror = False
