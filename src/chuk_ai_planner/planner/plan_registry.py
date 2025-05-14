# src/chuk_ai_planner/planner/plan_registry.py
# planner
from chuk_ai_planner.store.memory import InMemoryGraphStore

# universal plan
from .universal_plan import UniversalPlan

class PlanRegistry:
    """
    Registry for storing and retrieving UniversalPlans
    """
    
    def __init__(self, storage_dir: str = "plans"):
        self.storage_dir = storage_dir
        self.graph_store = InMemoryGraphStore()
        self.plans = {}  # id -> UniversalPlan
        
        # Create directories if they don't exist
        import os
        os.makedirs(storage_dir, exist_ok=True)
    
    def register_plan(self, plan: UniversalPlan) -> str:
        """
        Register a plan with the registry
        Returns the plan ID
        """
        # Ensure the plan is saved to its graph
        if not plan._indexed:
            plan.save()
        
        # Store the plan in memory
        self.plans[plan.id] = plan
        
        # Save the plan to disk
        self._save_plan_to_disk(plan)
        
        return plan.id
    
    def get_plan(self, plan_id: str) -> Optional[UniversalPlan]:
        """Get a plan by ID"""
        # Check in-memory cache
        if plan_id in self.plans:
            return self.plans[plan_id]
        
        # Try to load from disk
        plan = self._load_plan_from_disk(plan_id)
        if plan:
            self.plans[plan_id] = plan
            return plan
            
        return None
    
    def find_plans(self, tags: List[str] = None, title_contains: str = None) -> List[UniversalPlan]:
        """Find plans by tags and/or title"""
        # Load all plans if not in memory
        self._load_all_plans()
        
        # Filter plans
        result = []
        for plan in self.plans.values():
            # Filter by tags
            if tags and not any(tag in plan.tags for tag in tags):
                continue
                
            # Filter by title
            if title_contains and title_contains.lower() not in plan.title.lower():
                continue
                
            result.append(plan)
            
        return result
    
    def _save_plan_to_disk(self, plan: UniversalPlan) -> None:
        """Save a plan to disk"""
        import os
        import json
        
        # Convert plan to dictionary
        plan_dict = plan.to_dict()
        
        # Save to file
        file_path = os.path.join(self.storage_dir, f"{plan.id}.json")
        with open(file_path, 'w') as f:
            json.dump(plan_dict, f, indent=2)
    
    def _load_plan_from_disk(self, plan_id: str) -> Optional[UniversalPlan]:
        """Load a plan from disk"""
        import os
        import json
        
        file_path = os.path.join(self.storage_dir, f"{plan_id}.json")
        if not os.path.exists(file_path):
            return None
            
        try:
            with open(file_path, 'r') as f:
                plan_dict = json.load(f)
                
            # Create plan from dictionary
            return UniversalPlan.from_dict(plan_dict, graph=self.graph_store)
        except Exception as e:
            print(f"Error loading plan {plan_id}: {e}")
            return None
    
    def _load_all_plans(self) -> None:
        """Load all plans from disk"""
        import os
        
        for filename in os.listdir(self.storage_dir):
            if not filename.endswith('.json'):
                continue
                
            plan_id = filename[:-5]  # Remove .json
            if plan_id not in self.plans:
                self._load_plan_from_disk(plan_id)