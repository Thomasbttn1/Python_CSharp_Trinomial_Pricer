# node.py
class Node:
    """
    Nœud d'un arbre trinomial recombinant sans containers.
    Liaison verticale : down, mid, up (vers la colonne i+1)
    Liaison horizontale : left, right (dans la même colonne i)
    Les probabilités "ex-div" du step sortant sont portées par le nœud source : pd, pm, pu.
    """

    # ------------------------------------------------------------------
    # __slots__ :
    #   - Permet de définir explicitement la liste des attributs autorisés
    #     pour la classe, au lieu de créer dynamiquement un dictionnaire
    #     (__dict__) pour chaque instance.
    #   - Avantages :
    #       • Réduction importante de la mémoire (chaque Node est plus léger),
    #         ce qui est crucial dans un arbre trinomial où le nombre de nœuds
    #         croît en O(N²).
    #       • Accès plus rapide aux attributs.
    #       • Empêche l’ajout accidentel de nouveaux attributs (typos, etc.).
    #   - Inconvénient :
    #       • Impossible d’ajouter dynamiquement de nouveaux attributs
    #         non listés ici.
    # ------------------------------------------------------------------
    
    __slots__ = (
        "tree", "level", "index",
        "under", "forward", "value",
        "down", "mid", "up",
        "left", "right",
        "pd", "pm", "pu"
    )

    def __init__(self, tree, level: int, index: int):
        self.tree = tree
        self.level = level   # i
        self.index = index   # j (facultatif mais utile pour debug/plots)
        self.under = None    # S(i,j)
        self.forward = None  # forward cible (au "mid" choisi) si utile
        self.value = None    # valeur d'option pour le backward

        # enfants (colonne i+1)
        self.down = None
        self.mid  = None
        self.up   = None

        # fratrie (même colonne i)
        self.left  = None
        self.right = None

        # probas locales pour le step sortant SI ex-div (sinon None -> on utilise la triplette globale)
        self.pd = None
        self.pm = None
        self.pu = None

    # Petites aides d'itération
    def iter_right(self):
        cur = self
        while cur is not None:
            yield cur
            cur = cur.right

    def is_leaf(self):
        return (self.down is None) and (self.mid is None) and (self.up is None)




    




    
    
