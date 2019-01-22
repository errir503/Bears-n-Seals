class HotSpotMap:
    def __init__(self):
        self.images = {}
        self.hs_id_to_idx = {}
        self.hotspots = []
        return

    def add(self, hotspot):
        rgb = hotspot.rgb
        if rgb.path not in self.images:
            self.images[rgb.path] = []

        thermal = hotspot.thermal
        if thermal.path not in self.images:
            self.images[thermal.path] = []

        ir = hotspot.ir
        if ir.path not in self.images:
            self.images[ir.path] = []

        self.images[rgb.path].append(len(self.hotspots))
        self.images[thermal.path].append(len(self.hotspots))
        self.images[ir.path].append(len(self.hotspots))

        self.hs_id_to_idx[hotspot.id] = len(self.hotspots)
        self.hotspots.append(hotspot)
        return

    def get_hs(self, id):
        if str(id) in self.hs_id_to_idx:
            return self.hotspots[self.hs_id_to_idx[str(id)]]
        print("No HotSpot with id: " + str(id))
        return None

