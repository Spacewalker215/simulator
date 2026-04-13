using UnityEngine;

public class BrokenCarRandomizer : MonoBehaviour
{
    // This allows any script (like TcpCarHandler) to find this script instantly
    public static BrokenCarRandomizer Instance;

    private Vector3[] possibleLocations = new Vector3[]
    {
        new Vector3(20.6f, 1.02f, -14.1f),   // Location 1
        new Vector3(-1.54f, 1.02f, -18.1f),  // Location 2
        new Vector3(16.88f, 1.02f, 17.57f),  // Location 3
        new Vector3(-9.32f, 1.02f, 11.12f),  // Location 4
        new Vector3(50.82f, 1.02f, 16.04f)   // Location 5
    };

    void Awake()
    {
        // Set the singleton reference when the scene starts
        Instance = this;
        RandomizeLocation();
    }

    public void RandomizeLocation()
    {
        int randomIndex = Random.Range(0, possibleLocations.Length);
        transform.position = possibleLocations[randomIndex];
        // Debug.Log($"[RADAR] Reset Detected: Moving broken car to Location {randomIndex}");
    }
}
