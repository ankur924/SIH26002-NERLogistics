import { useState } from "react";

import {
  MapContainer,
  TileLayer,
  Marker,
  Popup,
  Polyline,
  CircleMarker,
} from "react-leaflet";

import "leaflet/dist/leaflet.css";

function Map() {
  const center = [26.2006, 92.9376];

  const locations = {
    Guwahati: [26.1445, 91.7362],
    Imphal: [24.8170, 93.9368],
    Shillong: [25.5788, 91.8933],
    Aizawl: [23.7271, 92.7176],
    Agartala: [23.8315, 91.2868],
    Kohima: [25.6751, 94.1086],
    Gangtok: [27.3389, 88.6065],
  };

  // -----------------------------
  // SIMULATED DISASTER DATA
  // -----------------------------

  const hazards = [
    {
      id: 1,
      lat: 25.5,
      lng: 93.2,
      type: "Landslide",
      severity: 9,
      description: "Heavy landslide risk on this corridor",
    },
    {
      id: 2,
      lat: 25.1,
      lng: 93.5,
      type: "Flood",
      severity: 8,
      description: "Flooding reported near the road",
    },
    {
      id: 3,
      lat: 26.0,
      lng: 92.4,
      type: "Heavy Rain",
      severity: 6,
      description: "Heavy rainfall may reduce road accessibility",
    },
    {
      id: 4,
      lat: 25.8,
      lng: 94.0,
      type: "Road Blockage",
      severity: 10,
      description: "Road temporarily blocked",
    },
  ];

  const [from, setFrom] = useState("Guwahati");
  const [to, setTo] = useState("Imphal");

  const [routes, setRoutes] = useState([]);
  const [loading, setLoading] = useState(false);

  // --------------------------------
  // DISTANCE BETWEEN TWO POINTS
  // --------------------------------

  const distanceBetweenPoints = (lat1, lng1, lat2, lng2) => {
    const dx = lat1 - lat2;
    const dy = lng1 - lng2;

    return Math.sqrt(dx * dx + dy * dy);
  };

  // --------------------------------
  // CALCULATE RISK OF A ROUTE
  // --------------------------------

  const calculateRouteRisk = (coordinates) => {
    let risk = 0;

    hazards.forEach((hazard) => {
      let closestDistance = Infinity;

      // Check every point of the route
      coordinates.forEach(([lat, lng]) => {
        const distance = distanceBetweenPoints(
          lat,
          lng,
          hazard.lat,
          hazard.lng
        );

        if (distance < closestDistance) {
          closestDistance = distance;
        }
      });

      // If route comes close to hazard
      if (closestDistance < 0.25) {
        risk += hazard.severity;
      } else if (closestDistance < 0.5) {
        risk += hazard.severity * 0.5;
      }
    });

    return Math.round(risk);
  };

  // --------------------------------
  // FIND ROUTE
  // --------------------------------

  const findRoute = async () => {
    if (from === to) {
      alert("Origin and destination cannot be the same.");
      return;
    }

    setLoading(true);

    const start = locations[from];
    const end = locations[to];

    const url =
      `https://router.project-osrm.org/route/v1/driving/` +
      `${start[1]},${start[0]};${end[1]},${end[0]}` +
      `?overview=full&geometries=geojson&alternatives=true`;

    try {
      const response = await fetch(url);
      const data = await response.json();

      if (data.code !== "Ok") {
        throw new Error("Route could not be found.");
      }

      const routeData = data.routes.map((route) => {
        const coordinates =
          route.geometry.coordinates.map(
            ([lng, lat]) => [lat, lng]
          );

        const riskScore = calculateRouteRisk(coordinates);

        return {
          coordinates: coordinates,
          distance: (route.distance / 1000).toFixed(1),
          duration: (route.duration / 3600).toFixed(1),
          riskScore: riskScore,
        };
      });

      // --------------------------------
      // FIND SAFEST PRACTICAL ROUTE
      // --------------------------------

      const scoredRoutes = routeData.map((route) => {
        const time = parseFloat(route.duration);

        // Risk has a strong penalty
        const score = time + route.riskScore * 0.1;

        return {
          ...route,
          score: score,
        };
      });

      const safestRoute = Math.min(
        ...scoredRoutes.map((route) => route.score)
      );

      const finalRoutes = scoredRoutes.map((route) => ({
        ...route,
        recommended: route.score === safestRoute,
      }));

      setRoutes(finalRoutes);

    } catch (error) {
      console.error(error);
      alert("Could not calculate route.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>

      {/* -------------------------------- */}
      {/* ROUTE CONTROLS */}
      {/* -------------------------------- */}

      <div
        style={{
          padding: "15px",
          display: "flex",
          gap: "15px",
          alignItems: "center",
          flexWrap: "wrap",
        }}
      >

        <label>
          From:

          <select
            value={from}
            onChange={(e) => setFrom(e.target.value)}
            style={{
              marginLeft: "5px",
              padding: "8px",
            }}
          >
            {Object.keys(locations).map((city) => (
              <option key={city} value={city}>
                {city}
              </option>
            ))}
          </select>
        </label>


        <label>
          To:

          <select
            value={to}
            onChange={(e) => setTo(e.target.value)}
            style={{
              marginLeft: "5px",
              padding: "8px",
            }}
          >
            {Object.keys(locations).map((city) => (
              <option key={city} value={city}>
                {city}
              </option>
            ))}
          </select>
        </label>


        <button
          onClick={findRoute}
          disabled={loading}
          style={{
            padding: "9px 16px",
            cursor: "pointer",
          }}
        >
          {loading ? "Analyzing Routes..." : "Find Safe Route"}
        </button>

      </div>


      {/* -------------------------------- */}
      {/* ROUTE RESULTS */}
      {/* -------------------------------- */}

      {routes.length > 0 && (

        <div style={{ padding: "0 15px 15px" }}>

          <h3>AI Route Analysis</h3>

          {routes.map((route, index) => (

            <div
              key={index}
              style={{
                border: route.recommended
                  ? "3px solid green"
                  : "1px solid #ccc",

                padding: "12px",
                marginBottom: "10px",
                borderRadius: "8px",

                backgroundColor: route.recommended
                  ? "#eaffea"
                  : "white",
              }}
            >

              <strong>
                Route {index + 1}
              </strong>

              {route.recommended && (
                <span
                  style={{
                    marginLeft: "10px",
                    color: "green",
                    fontWeight: "bold",
                  }}
                >
                  ✓ RECOMMENDED
                </span>
              )}

              <p>
                Distance: {route.distance} km
                <br />

                Estimated time: {route.duration} hours
                <br />

                Risk Score: {route.riskScore}/100
              </p>

              {route.riskScore >= 15 && (
                <strong style={{ color: "red" }}>
                  ⚠ High Risk Route
                </strong>
              )}

              {route.riskScore > 0 &&
                route.riskScore < 15 && (
                  <strong style={{ color: "orange" }}>
                    ⚠ Moderate Risk
                  </strong>
                )}

              {route.riskScore === 0 && (
                <strong style={{ color: "green" }}>
                  ✓ Low Risk
                </strong>
              )}

            </div>

          ))}

        </div>

      )}


      {/* -------------------------------- */}
      {/* MAP */}
      {/* -------------------------------- */}

      <MapContainer
        center={center}
        zoom={6}
        style={{
          height: "500px",
          width: "100%",
        }}
      >

        <TileLayer
          attribution="&copy; OpenStreetMap contributors"
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />


        {/* ORIGIN */}

        <Marker position={locations[from]}>
          <Popup>
            Origin: {from}
          </Popup>
        </Marker>


        {/* DESTINATION */}

        <Marker position={locations[to]}>
          <Popup>
            Destination: {to}
          </Popup>
        </Marker>


        {/* -------------------------------- */}
        {/* DISASTER / HAZARD MARKERS */}
        {/* -------------------------------- */}

        {hazards.map((hazard) => (

          <CircleMarker
            key={hazard.id}
            center={[hazard.lat, hazard.lng]}
            radius={12}
            pathOptions={{
              fillOpacity: 0.8,
            }}
          >

            <Popup>

              <strong>
                {hazard.type}
              </strong>

              <br />

              Severity: {hazard.severity}/10

              <br />

              {hazard.description}

            </Popup>

          </CircleMarker>

        ))}


        {/* -------------------------------- */}
        {/* ROUTES */}
        {/* -------------------------------- */}

        {routes.map((route, index) => (

          <Polyline
            key={index}
            positions={route.coordinates}

            pathOptions={{
              weight: route.recommended ? 7 : 4,

              opacity: route.recommended ? 1 : 0.5,
            }}
          />

        ))}

      </MapContainer>

    </div>
  );
}

export default Map;