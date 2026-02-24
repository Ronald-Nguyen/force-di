public abstract class di_Binding implements Comparable {

    public BindingType BindingType {get; private set;}

    @AuraEnabled
    public String BindingTypeAsString {get { return BindingType.name();} }

    @AuraEnabled
    public String NamespacePrefix {get;private set;}

    @AuraEnabled
    public String DeveloperName {get;private set;}

    @AuraEnabled
    public SObjectType BindingObject {get;private set;}

    @AuraEnabled
    public Integer BindingSequence {get;private set;}

    @AuraEnabled
    public Object To {get;private set;}

    @AuraEnabled
    public Object Data {get;private set;}

    private Boolean IsProvider = false;
    private Object Injected = null;

    
    public Object getInstance() {
        return getInstance(null, false);
    }

    
    public Object getInstance(Object params) {
        return getInstance(params, false);
    }

    
    public Object getInstance(Object params, Boolean newInstance) {
        if(IsProvider) {
            return newInstance(params);
        } 
        if(newInstance == true) {
            return newInstance(params);
        } 
        if(Injected == null) {
            Injected = newInstance(params);
        }
        return Injected;
    }

    
    public abstract Object newInstance(Object params);

    
    public Integer compareTo(Object compareTo) {
        di_Binding binding = (di_Binding) compareTo;
        return this.toString().compareTo(binding.toString());
    }

    
    public override String toString() {
        String hashValue = 
            (BindingObject != null) ? 
            BindingObject.getDescribe().getName() : 
            DeveloperName;
        if(BindingSequence != null) {
            hashValue += '#' + String.valueOf(BindingSequence).leftPad(4,'0');
        }
        return hashValue;
    }

    
    public interface Provider {
        Object newInstance(Object params);
    }

    
    public class Resolver {

        private String developerName;
        private SObjectType bindingObject;
        private boolean bindingsAreRequired = true;
        
        private List<di_Module> modules = null;
        private List<di_Binding> bindings = null;

        public Resolver(List<di_Module> modules) {
            this.modules = modules;
        }

        public Resolver set(di_Module module) {
            modules.clear();
            modules.add(module);
            bindings = null; 
            return this;
        }

        public Resolver add(di_Module module) {
            modules.add(module);
            bindings = null; 
            return this;
        }

        public Resolver byName(String developerName) {
            this.developerName = developerName;
            return this;
        }

        public Resolver bySObject(SObjectType bindingObject) {
            this.bindingObject = bindingObject;
            return this;
        }

        public Resolver emptyBindingsAllowed() {
            bindingsAreRequired = false;
            return this;
        }

        public Resolver replaceBindingWith( Object mockType )
        {
            loadBindings();
            for (Integer i = 0; i < this.bindings.size(); i++) 
            {
                if ( isBindingMatchByFilteringCriteria( this.bindings[i] ) )
                {
                    this.bindings[i] = di_Binding.newInstance(di_Binding.BindingType.Apex, this.developerName, this.bindingObject, null, mockType, null);
                    break;
                }
            }
            this.developerName = null;
            this.bindingObject = null;
            return this;
        }

        private Boolean isBindingMatchByFilteringCriteria( di_Binding bind )
        {
            Boolean nameMatch = isNameMatch(bind);
            
            if ( String.isNotBlank(this.developerName) && this.bindingObject != null ) {
                return nameMatch && (this.bindingObject == bind.BindingObject);
            } 
            
            if ( String.isNotBlank(this.developerName) ) {
                return nameMatch;
            } 
            
            if ( this.bindingObject != null ) {
                return this.bindingObject == bind.BindingObject;
            }
            
            return false;
        }

        private Boolean isNameMatch(di_Binding bind) {
            if (String.isBlank(this.developerName)) {
                return false;
            }
            return this.developerName.equalsIgnoreCase(bind.DeveloperName) 
                || this.developerName.equalsIgnoreCase(di_NamespaceClass.CURRENTNAMESPACE + '.' + bind.DeveloperName);
        }

        private void loadBindings()
        {
            if ( this.bindings != null ) {
                return;
            }
            this.bindings = new List<di_Binding>();
            for (di_Module module : modules) 
            {
                module.configure();
                for(di_Binding bind : module.getBindings()) {
                    if(bind.BindingType == di_Binding.BindingType.Module) {
                        di_Module embeddedModule = (di_Module) bind.getInstance();
                        embeddedModule.configure();
                        this.bindings.addAll(embeddedModule.getBindings());
                    } else {
                        this.bindings.add(bind);
                    }
                }
            }
        }

        public List<di_Binding> get() 
        {
            list<di_Binding> matchedBindings = di_PlatformCache.getInstance().retrieveBindings(this.developerName, this.bindingObject);
            Boolean cacheEnabled = di_PlatformCache.isStoringBindingInPlatformCache();
            
            if ( !cacheEnabled || (bindingsAreRequired && matchedBindings.isEmpty()) )
            {
                loadBindings();
                for (di_Binding bind : bindings) 
                {
                    if ( isBindingMatchByFilteringCriteria(bind) )
                    {
                        matchedBindings.add(bind);
                    }
                }
            }

            this.developerName = null;
            this.bindingObject = null;
            matchedBindings.sort();
            this.bindingsAreRequired = true;
            
            return matchedBindings;
        }
    }

    public class BindingException extends Exception {}

    public enum BindingType { Apex, VisualforceComponent, LightningComponent, Flow, Module }

    public static di_Binding newInstance(
            BindingType bindType,
            String developerName,
            SObjectType bindingObject,
            Integer bindingSequence,
            Object to,
            Object bindingData) 
    {
        Type implType = BINDING_IMPLS_BY_TYPE.get(bindType);

        if ( implType != null ) {
            di_Binding binding = (di_Binding) implType.newInstance();
            binding.BindingType = bindType;
            binding.DeveloperName = developerName;
            binding.BindingObject = bindingObject;
            binding.BindingSequence = bindingSequence;
            binding.To = to;
            binding.Data = bindingData;
            return binding;
        }
        throw new BindingException('Binding type ' + bindType + ' has no implementation.');
    }

    private static final Map<BindingType, Type> BINDING_IMPLS_BY_TYPE =
        new Map<BindingType, Type> {
            BindingType.Apex => ApexBinding.class,
            BindingType.LightningComponent => LightningComponentBinding.class,
            BindingType.VisualforceComponent => VisualForceComponentBinding.class,
            BindingType.Flow => FlowBinding.class,
            BindingType.Module => ApexBinding.class
        };

    private class ApexBinding extends di_Binding 
    {
        public override Object newInstance(Object params) 
        {
            if ( !(To instanceof String) ) {
                return To;
            }
            
            String className = (String) To;
            Type toType = NamespacePrefix == null ? Type.forName(className) : Type.forName(NamespacePrefix, className);
            if ( toType == null ) 
            {
                throw new BindingException('Apex binding ' + DeveloperName + ' implementation ' + To + ' does not exist');
            }
            Object toObject = toType.newInstance();
            
            IsProvider = toObject instanceof Provider;
            if ( IsProvider )
            {
                return ((Provider) toObject).newInstance(params);
            } 
            
            if ( params != null ) 
            {
                throw new BindingException('Apex binding ' + DeveloperName + ' implementation ' + className + ' does not implement the Provider interface.');
            }
            return toObject;
        }
    }

    private class VisualForceComponentBinding extends di_Binding 
    {
        public  override Object newInstance(Object params) 
        {
            if ( !(To instanceof String) ) {
                return To;
            }
                
            String className = (String) To;
            Type toType = NamespacePrefix == null ? Type.forName(className) : Type.forName(NamespacePrefix, className);
            if ( toType == null )
            {
                throw new BindingException('Visualforce Component binding ' + DeveloperName + ' implementation ' + className + ' does not exist.');
            }
            
            Object toObject = toType.newInstance();
            IsProvider = toObject instanceof Provider;
            if ( IsProvider )
            {
                return ((Provider) toObject).newInstance(params);
            }
            throw new BindingException('Visualforce Component binding ' + DeveloperName + ' must point to a class implementing the Provider interface.');
        }
    }

    private class LightningComponentBinding extends di_Binding 
    {
        public override Object newInstance(Object params) 
        {
            return To;
        }
    }

    private class FlowBinding extends di_Binding 
    {
        public  override Object newInstance(Object params) 
        {
            if ( !(To instanceof String) ) {
                return To;
            }
            String flowName = (String) To;
            Map<String, Object> inputParams = (params instanceof Map<String, Object>) ? (Map<String, Object>) params : new Map<String, Object>();
            return new di_Flow(Flow.Interview.createInterview(flowName, inputParams));
        }
    }
}