"""
Unit tests for the Maximum Visibility Time (MVT) algorithm.

This module tests various aspects of the MVT algorithm to ensure
it correctly assigns commodities to satellites while minimizing handovers.
"""

import unittest
import os
import tempfile
from collections import defaultdict
from typing import Dict, List

from satellite_scheduling.algorithms.mvt import (
    MVT, calculate_visible_time, count_handovers, write_mvt_results_to_file
)


class TestMVT(unittest.TestCase):
    """Test cases for the MVT algorithm."""

    def setUp(self):
        """Set up test cases with sample data."""
        # Simple satellite timeline
        self.sat_timeline = [
            ["S1", "S2", "S3"],         # t=0: Satellites S1, S2, S3 visible
            ["S1", "S2", "S4"],         # t=1: S3 disappears, S4 appears
            ["S1", "S4", "S5"],         # t=2: S2 disappears, S5 appears
            ["S1", "S5", "S6"]          # t=3: S4 disappears, S6 appears
        ]
        
        # Satellite capacities
        self.satellite_capacity = {
            "S1": 10,
            "S2": 8,
            "S3": 5,
            "S4": 7,
            "S5": 9,
            "S6": 6
        }
        
        # Commodity demands
        self.commodity_demands = {
            "C1": 5,
            "C2": 8,
            "C3": 3
        }
        
        # Simple case for handover counting
        self.simple_assignments = {
            0: {"C1": {"S1": 5}, "C2": {"S2": 8}},
            1: {"C1": {"S1": 5}, "C2": {"S4": 8}},  # C2 handover: S2 -> S4
            2: {"C1": {"S1": 5}, "C2": {"S4": 8}},
            3: {"C1": {"S1": 5}, "C2": {"S5": 8}}   # C2 handover: S4 -> S5
        }
        
        # Create temporary files for test outputs
        self.temp_path_file = tempfile.NamedTemporaryFile(delete=False).name
        self.temp_ho_file = tempfile.NamedTemporaryFile(delete=False).name

    def tearDown(self):
        """Clean up temporary files after tests."""
        try:
            os.remove(self.temp_path_file)
            os.remove(self.temp_ho_file)
        except:
            pass

    def test_calculate_visible_time(self):
        """Test the calculation of satellite visibility duration."""
        # S1 is visible for all 4 time slots starting from t=0
        self.assertEqual(calculate_visible_time(self.sat_timeline, 0, "S1"), 4)
        
        # S2 is visible for 2 time slots starting from t=0
        self.assertEqual(calculate_visible_time(self.sat_timeline, 0, "S2"), 2)
        
        # S3 is visible for only 1 time slot starting from t=0
        self.assertEqual(calculate_visible_time(self.sat_timeline, 0, "S3"), 1)
        
        # S4 is visible for 2 time slots starting from t=1
        self.assertEqual(calculate_visible_time(self.sat_timeline, 1, "S4"), 2)
        
        # S5 is visible for 2 time slots starting from t=2
        self.assertEqual(calculate_visible_time(self.sat_timeline, 2, "S5"), 2)
        
        # Edge case: non-existent satellite
        self.assertEqual(calculate_visible_time(self.sat_timeline, 0, "S7"), 0)

    def test_count_handovers(self):
        """Test the handover counting function."""
        handover_counts = count_handovers(self.simple_assignments)
        
        # C1 should have 0 handovers (stays with S1)
        self.assertEqual(handover_counts.get("C1", 0), 0)
        
        # C2 should have 2 handovers (S2 -> S4 -> S5)
        self.assertEqual(handover_counts.get("C2", 0), 2)
        
        # C3 is not in assignments, should have 0 handovers
        self.assertEqual(handover_counts.get("C3", 0), 0)
        
        # Total handovers should be 2
        self.assertEqual(sum(handover_counts.values()), 2)

    def test_write_mvt_results_to_file(self):
        """Test writing MVT results to output files."""
        # Write results to temporary files
        write_mvt_results_to_file(self.simple_assignments, self.temp_path_file, self.temp_ho_file)
        
        # Check that files were created
        self.assertTrue(os.path.exists(self.temp_path_file))
        self.assertTrue(os.path.exists(self.temp_ho_file))
        
        # Check handover count file content
        with open(self.temp_ho_file, 'r') as f:
            handover_count = int(f.read().strip())
            self.assertEqual(handover_count, 2)
        
        # Check that path file contains expected information
        with open(self.temp_path_file, 'r') as f:
            content = f.read()
            self.assertIn("Total number of handovers: 2", content)
            self.assertIn("Commodity C1: 0 handovers", content)
            self.assertIn("Commodity C2: 2 handovers", content)

    def test_mvt_basic(self):
        """Test basic functionality of the MVT algorithm."""
        # Run the MVT algorithm
        assignments = MVT(
            self.sat_timeline, 
            self.satellite_capacity, 
            self.commodity_demands, 
            self.temp_path_file, 
            self.temp_ho_file
        )
        
        # Check that assignments were made for all time slots
        self.assertEqual(len(assignments), len(self.sat_timeline))
        
        # Check that all commodities are assigned in the first time slot
        self.assertEqual(len(assignments[0]), len(self.commodity_demands))
        
        # Verify assignments follow satellite visibility
        for t, time_assignments in assignments.items():
            for comm_id, sat_dict in time_assignments.items():
                for sat in sat_dict.keys():
                    # Satellite must be visible at this time
                    self.assertIn(sat, self.sat_timeline[t])

    def test_mvt_capacity_constraint(self):
        """Test that MVT respects satellite capacity constraints."""
        # Run the MVT algorithm
        assignments = MVT(
            self.sat_timeline, 
            self.satellite_capacity, 
            self.commodity_demands, 
            self.temp_path_file, 
            self.temp_ho_file
        )
        
        # Check that satellite capacities are not exceeded
        for t, time_assignments in assignments.items():
            satellite_usage = defaultdict(int)
            
            for comm_id, sat_dict in time_assignments.items():
                for sat, flow in sat_dict.items():
                    satellite_usage[sat] += flow
            
            for sat, usage in satellite_usage.items():
                self.assertLessEqual(usage, self.satellite_capacity[sat])

    def test_mvt_demand_satisfaction(self):
        """Test that MVT satisfies all commodity demands."""
        # Run the MVT algorithm
        assignments = MVT(
            self.sat_timeline, 
            self.satellite_capacity, 
            self.commodity_demands, 
            self.temp_path_file, 
            self.temp_ho_file
        )
        
        # Check that all demands are met in each time slot
        for t, time_assignments in assignments.items():
            for comm_id, sat_dict in time_assignments.items():
                total_flow = sum(sat_dict.values())
                self.assertEqual(total_flow, self.commodity_demands[comm_id])

    '''
    def test_mvt_handover_minimization(self):
        """Test that MVT effectively minimizes handovers."""
        # Run the MVT algorithm
        assignments = MVT(
            self.sat_timeline, 
            self.satellite_capacity, 
            self.commodity_demands, 
            self.temp_path_file, 
            self.temp_ho_file
        )
        
        # Count handovers
        handover_counts = count_handovers(assignments)
        total_handovers = sum(handover_counts.values())
        
        # S1 is visible throughout, so C1 (demand 5) should have 0 handovers
        self.assertEqual(handover_counts.get("C1", 0), 0)
        
        # Check that handovers are minimized for other commodities
        # Since we can't know exactly how many handovers are optimal without checking all possibilities,
        # we'll check that the number is reasonably low based on the satellite timeline
        self.assertLessEqual(total_handovers, len(self.commodity_demands) * 2)
    '''

    def test_mvt_visibility_prioritization(self):
        """Test that MVT prioritizes satellites with longer visibility."""
        # Create a specific test case where S1 has long visibility but low capacity
        sat_timeline = [
            ["S1", "S2"],  # t=0
            ["S1", "S2"],  # t=1
            ["S1", "S3"],  # t=2
            ["S1", "S3"]   # t=3
        ]
        
        satellite_capacity = {
            "S1": 3,  # Low capacity but visible throughout
            "S2": 10,  # High capacity, visible for t=0,1
            "S3": 10   # High capacity, visible for t=2,3
        }
        
        commodity_demands = {
            "C1": 3,  # Exactly matches S1's capacity
            "C2": 10  # Needs full capacity of S2 or S3
        }
        
        # Run the MVT algorithm
        assignments = MVT(
            sat_timeline, 
            satellite_capacity, 
            commodity_demands, 
            self.temp_path_file, 
            self.temp_ho_file
        )
        
        # C1 should be assigned to S1 throughout due to longest visibility
        for t in range(len(sat_timeline)):
            self.assertIn("C1", assignments[t])
            self.assertIn("S1", assignments[t]["C1"])
        
        # C2 should be assigned to S2 for t=0,1 and S3 for t=2,3 to minimize handovers
        self.assertIn("S2", assignments[0]["C2"])
        self.assertIn("S2", assignments[1]["C2"])
        self.assertIn("S3", assignments[2]["C2"])
        self.assertIn("S3", assignments[3]["C2"])
        
        # Count handovers - C2 should have exactly 1 handover
        handover_counts = count_handovers(assignments)
        self.assertEqual(handover_counts.get("C2", 0), 1)


if __name__ == '__main__':
    unittest.main()

    