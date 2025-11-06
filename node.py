# node.py  # définition d’un nœud d’arbre

class Node:  # nœud d'un arbre trinomial recombinant
    """
    Nœud d'un arbre trinomial recombinant sans containers.
    Liaison verticale : down, mid, up (vers la colonne i+1)
    Liaison horizontale : left, right (dans la même colonne i)
    Les probabilités "ex-div" du step sortant sont portées par le nœud source : pd, pm, pu.
    """
    # __slots__ pour limiter la mémoire et accélérer l’accès  # explication courte
    __slots__ = (  # liste des attributs autorisés
        "tree", "level", "index",  # contexte et position
        "under", "forward", "value",  # données de nœud
        "down", "mid", "up",  # enfants (colonne i+1)
        "left", "right",  # voisins horizontaux (colonne i)
        "pd", "pm", "pu",  # probabilités locales
        "proba_reach"  # probabilité d’atteinte (pruning)
    )

    def __init__(self, tree, level: int, index: int):  # constructeur
        self.tree = tree  # référence vers l’arbre parent
        self.level = level   # i
        self.index = index   # j (utile pour debug/plots)
        self.under = None    # S(i,j)
        self.forward = None  # forward cible éventuel
        self.value = None    # valeur d'option (backward)

        self.down = None  # enfant bas
        self.mid  = None  # enfant milieu
        self.up   = None  # enfant haut

        self.left  = None  # voisin gauche
        self.right = None  # voisin droit

        self.pd = None  # proba down locale
        self.pm = None  # proba mid locale
        self.pu = None  # proba up locale

        self.proba_reach = 0.0  # proba d’atteinte pour pruning

    def iter_right(self):  # itère de ce nœud vers la droite
        cur = self  # départ sur le nœud courant
        while cur is not None:  # boucle jusqu’au bout de la fratrie
            yield cur  # renvoie le nœud courant
            cur = cur.right  # avance d’un voisin à droite

    def is_leaf(self):  # vrai si aucun enfant
        return (self.down is None) and (self.mid is None) and (self.up is None)  #











