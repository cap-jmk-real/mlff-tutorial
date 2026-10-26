class Box:
    """
    Simple class to define the unit box for the solvation

    """

    def __init__(self, lenght, height, widht, molecules):

        self.lenght = lenght
        self.height = height
        self.widht = widht
        self.molecules.append(*molecules)

    def get_boundaries(self):

        return [self.lenght, self.height, self.widht]

    def get_volume(self):
        """
        Program is built in Angstroms
        """
        return self.lenght * self.height * self.widht

    def make_xyz_block(self):
        raise NotImplementedError
