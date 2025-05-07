import mango_db as db
import mango_rag as rag

class MangoChat(object):
    """
    Master-Class governing the RAG "MangoChat".
    
    Users should interact directly only with this class. The
    MangoChat class, in turn, calls the different classes
    defined in mango_db.py and mango_rag.py in order to 
    generate the final answer.
    """
    
    def __init__(self):
        pass
    
    def start(self):
        """
        Start MangoChat session.
        """
        pass
        
    def stop(self):
        """
        End MangoChat session.
        """
        pass