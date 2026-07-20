
import numpy as np
from geographiclib.geodesic import Geodesic
from common.rotations import Rotation

class ECEF:
    @staticmethod
    def ecef_to_ned(lat_deg, lon_deg):
        lat, lon = np.radians([lat_deg, lon_deg])
        return Rotation.Y(np.pi / 2 + lat) @ Rotation.Z(-lon)
    
    @staticmethod
    def find_location(ecef_vectors : np.array, ned_vectors : np.array):
        rotation = Rotation.align(
            ecef_vectors,
            ned_vectors
        )

        x, y, z = -rotation[2]

        latitude = np.degrees(np.arctan2(z, np.hypot(x, y)))
        longitude = np.degrees(np.arctan2(y, x))

        return latitude, longitude
    
    def north_east_vector(lat1, lon1, lat2, lon2):
        result = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)

        distance = result["s12"]
        azimuth = np.radians(result["azi1"])

        return distance * np.array([
            np.cos(azimuth),
            np.sin(azimuth),
        ])

