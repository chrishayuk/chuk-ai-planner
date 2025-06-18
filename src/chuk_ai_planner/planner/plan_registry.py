# chuk_ai_planner/planner/plan_registry.py
"""
chuk_ai_planner.planner.plan_registry
====================================

Registry for storing and retrieving UniversalPlans.

This module provides a PlanRegistry class that enables:
- Registering plans
- Looking up plans by ID
- Finding plans by tags or title content
- Persisting plans to disk
- Loading plans from disk

Typical usage:
    registry = PlanRegistry()
    
    # Register a plan
    plan = UniversalPlan("My Plan")
    registry.register_plan(plan)
    
    # Get a plan by ID
    plan = registry.get_plan(plan_id)
    
    # Find plans by tags or title
    plans = registry.find_plans(tags=["research"], title_contains="climate")
"""

import os
import json
import logging
from typing import Dict, List, Optional, Union

# planner
from chuk_ai_planner.store.memory import InMemoryGraphStore

# universal plan
from .universal_plan import UniversalPlan

# Import serialization utility to handle frozen structures
from chuk_ai_planner.utils.serialization import unfreeze_data

_logger = logging.getLogger(__name__)

class PlanRegistry:
    """
    Registry for storing and retrieving UniversalPlans.
    
    This class provides methods for registering, retrieving, and searching
    for plans, as well as persisting them to disk.
    """
    
    def __init__(self, storage_dir: str = "plans"):
        """
        Initialize the plan registry.
        
        Parameters
        ----------
        storage_dir : str
            Directory where plans will be persisted
        """
        self.storage_dir = storage_dir
        self.graph_store = InMemoryGraphStore()
        self.plans = {}  # id -> UniversalPlan
        
        # Create storage directory if it doesn't exist
        os.makedirs(storage_dir, exist_ok=True)
        _logger.info(f"Initialized PlanRegistry with storage in '{storage_dir}'")
    
    def register_plan(self, plan: UniversalPlan) -> str:
        """
        Register a plan with the registry.
        
        Parameters
        ----------
        plan : UniversalPlan
            The plan to register
            
        Returns
        -------
        str
            The ID of the registered plan
        """
        # Ensure the plan is saved to its graph
        if not plan._indexed:
            plan.save()
        
        # Store the plan in memory
        self.plans[plan.id] = plan
        
        # Save the plan to disk
        self._save_plan_to_disk(plan)
        
        _logger.debug(f"Registered plan '{plan.title}' with ID {plan.id}")
        return plan.id
    
    def get_plan(self, plan_id: str) -> Optional[UniversalPlan]:
        """
        Get a plan by ID.
        
        Parameters
        ----------
        plan_id : str
            The ID of the plan to retrieve
            
        Returns
        -------
        Optional[UniversalPlan]
            The plan, or None if not found
        """
        # Check in-memory cache
        if plan_id in self.plans:
            return self.plans[plan_id]
        
        # Try to load from disk
        plan = self._load_plan_from_disk(plan_id)
        if plan:
            self.plans[plan_id] = plan
            return plan
        
        _logger.warning(f"Plan with ID {plan_id} not found")
        return None
    
    def find_plans(
        self, 
        tags: Optional[List[str]] = None, 
        title_contains: Optional[str] = None
    ) -> List[UniversalPlan]:
        """
        Find plans by tags and/or title content.
        
        Parameters
        ----------
        tags : Optional[List[str]]
            List of tags to search for (OR logic - any matching tag will include the plan)
        title_contains : Optional[str]
            String to search for in plan titles (case-insensitive)
            
        Returns
        -------
        List[UniversalPlan]
            List of matching plans
        """
        # Load all plans if not in memory
        self._load_all_plans()
        
        # Filter plans
        result = []
        for plan in self.plans.values():
            # Filter by tags (if specified)
            if tags and not any(tag in plan.tags for tag in tags):
                continue
                
            # Filter by title (if specified)
            if title_contains and title_contains.lower() not in plan.title.lower():
                continue
                
            result.append(plan)
        
        _logger.debug(f"Found {len(result)} plans matching criteria")
        return result
    
    def delete_plan(self, plan_id: str) -> bool:
        """
        Delete a plan from the registry and disk.
        
        Parameters
        ----------
        plan_id : str
            The ID of the plan to delete
            
        Returns
        -------
        bool
            True if the plan was deleted, False if not found
        """
        # Remove from memory
        if plan_id in self.plans:
            del self.plans[plan_id]
        else:
            _logger.warning(f"Plan {plan_id} not found in memory")
        
        # Remove from disk
        file_path = os.path.join(self.storage_dir, f"{plan_id}.json")
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                _logger.info(f"Deleted plan file {file_path}")
                return True
            except Exception as e:
                _logger.error(f"Failed to delete plan file {file_path}: {e}")
                return False
        else:
            _logger.warning(f"Plan file {file_path} not found")
            return False
    
    def get_all_plans(self) -> List[UniversalPlan]:
        """
        Get all plans in the registry.
        
        Returns
        -------
        List[UniversalPlan]
            List of all plans
        """
        self._load_all_plans()
        return list(self.plans.values())
    
    def _save_plan_to_disk(self, plan: UniversalPlan) -> None:
        """
        Save a plan to disk.
        
        Parameters
        ----------
        plan : UniversalPlan
            The plan to save
        """
        try:
            # Convert plan to dictionary
            plan_dict = plan.to_dict()
            
            # Use unfreeze_data to handle MappingProxyType and other frozen structures
            unfrozen_dict = unfreeze_data(plan_dict)
            
            # Save to file
            file_path = os.path.join(self.storage_dir, f"{plan.id}.json")
            with open(file_path, 'w') as f:
                json.dump(unfrozen_dict, f, indent=2)
                
            _logger.debug(f"Saved plan {plan.id} to {file_path}")
        except Exception as e:
            _logger.error(f"Error saving plan {plan.id} to disk: {e}")
    
    def _load_plan_from_disk(self, plan_id: str) -> Optional[UniversalPlan]:
        """
        Load a plan from disk.
        
        Parameters
        ----------
        plan_id : str
            The ID of the plan to load
            
        Returns
        -------
        Optional[UniversalPlan]
            The loaded plan, or None if not found or error occurred
        """
        file_path = os.path.join(self.storage_dir, f"{plan_id}.json")
        if not os.path.exists(file_path):
            _logger.debug(f"Plan file {file_path} not found")
            return None
            
        try:
            with open(file_path, 'r') as f:
                plan_dict = json.load(f)
                
            # Create plan from dictionary
            plan = UniversalPlan.from_dict(plan_dict, graph=self.graph_store)
            _logger.debug(f"Loaded plan {plan_id} from {file_path}")
            return plan
        except Exception as e:
            _logger.error(f"Error loading plan {plan_id} from disk: {e}")
            return None
    
    def _load_all_plans(self) -> None:
        """
        Load all plans from disk into memory.
        """
        try:
            loaded_count = 0
            for filename in os.listdir(self.storage_dir):
                if not filename.endswith('.json'):
                    continue
                    
                plan_id = filename[:-5]  # Remove .json
                if plan_id not in self.plans:
                    plan = self._load_plan_from_disk(plan_id)
                    if plan:
                        self.plans[plan_id] = plan
                        loaded_count += 1
                        
            if loaded_count > 0:
                _logger.debug(f"Loaded {loaded_count} plans from disk")
        except Exception as e:
            _logger.error(f"Error loading plans from disk: {e}")