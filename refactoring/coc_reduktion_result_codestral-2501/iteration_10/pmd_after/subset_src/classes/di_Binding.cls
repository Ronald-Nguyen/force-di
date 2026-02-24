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

        if(newInstance) {
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
        String hashValue = getBaseHashValue();
        if(BindingSequence != null) {
            hashValue += '#' + String.valueOf(BindingSequence).leftPad(4,'0');
        }
        return hashValue;
    }

    private String getBaseHashValue() {
        return (BindingObject != null) ?
            BindingObject.getDescribe().getName() :
            DeveloperName;
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

        public Resolver replaceBindingWith(Object mockType) {
            loadBindings();

            for (Integer currentBindingsIndex = 0; currentBindingsIndex < this.bindings.size(); currentBindingsIndex++) {
                if (isBindingMatchByFilteringCriteria(this.bindings[currentBindingsIndex])) {
                    this.bindings[currentBindingsIndex] = di_Binding.newInstance(
                        di_Binding.BindingType.Apex,
                        this.developerName,
                        this.bindingObject,
                        null,
                        mockType,
                        null
                    );
                    break;
                }
            }

            resetFilterCriteria();
            return this;
        }

        private void resetFilterCriteria() {
            this.developerName = null;
            this.bindingObject = null;
        }

        private Boolean isBindingMatchByFilteringCriteria(di_Binding bind) {
            if (isExactMatch(bind)) {
                return true;
            }

            if (isDeveloperNameMatch(bind)) {
                return true;
            }

            if (isBindingObjectMatch(bind)) {
                return true;
            }

            return false;
        }

        private Boolean isExactMatch(di_Binding bind) {
            return String.isNotBlank(this.developerName) &&
                   this.bindingObject != null &&
                   (this.developerName.equalsIgnoreCase(bind.DeveloperName) ||
                    this.developerName.equalsIgnoreCase(di_NamespaceClass.CURRENTNAMESPACE + '.' + bind.DeveloperName)) &&
                   this.bindingObject == bind.BindingObject;
        }

        private Boolean isDeveloperNameMatch(di_Binding bind) {
            return String.isNotBlank(this.developerName) &&
                   (this.developerName.equalsIgnoreCase(bind.DeveloperName) ||
                    this.developerName.equalsIgnoreCase(di_NamespaceClass.CURRENTNAMESPACE + '.' + bind.DeveloperName));
        }

        private Boolean isBindingObjectMatch(di_Binding bind) {
            return this.bindingObject != null &&
                   bind.BindingObject == this.bindingObject;
        }

        private void loadBindings() {
            if (this.bindings == null) {
                this.bindings = new List<di_Binding>();
                for (di_Module module : modules) {
                    processModule(module);
                }
            }
        }

        private void processModule(di_Module module) {
            module.configure();
            for(di_Binding bind : module.getBindings()) {
                if(isModuleBinding(bind)) {
                    processEmbeddedModule(bind);
                } else {
                    this.bindings.add(bind);
                }
            }
        }

        private Boolean isModuleBinding(di_Binding bind) {
            return bind.BindingType == di_Binding.BindingType.Module;
        }

        private void processEmbeddedModule(di_Binding bind) {
            di_Module embeddedModule = (di_Module) bind.getInstance();
            embeddedModule.configure();
            this.bindings.addAll(embeddedModule.getBindings());
        }

        public List<di_Binding> get() {
            List<di_Binding> matchedBindings = getCachedBindings();

            if (shouldLoadBindingsFromSource()) {
                loadBindingsFromSource(matchedBindings);
            }

            resetFilterCriteria();
            matchedBindings.sort();

            this.bindingsAreRequired = true;

            return matchedBindings;
        }

        private List<di_Binding> getCachedBindings() {
            return di_PlatformCache.getInstance().retrieveBindings(this.developerName, this.bindingObject);
        }

        private Boolean shouldLoadBindingsFromSource() {
            return (!di_PlatformCache.isStoringBindingInPlatformCache() && bindingsAreRequired) ||
                   (di_PlatformCache.isStoringBindingInPlatformCache() && bindingsAreRequired && matchedBindings.isEmpty());
        }

        private void loadBindingsFromSource(List<di_Binding> matchedBindings) {
            loadBindings();

            for (di_Binding bind : bindings) {
                if (isBindingMatchByFilteringCriteria(bind)) {
                    matchedBindings.add(bind);
                }
            }
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
            Object bindingData) {

        Type implType = BINDING_IMPLS_BY_TYPE.get(bindType);

        if (implType != null) {
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

    private class ApexBinding extends di_Binding {
        public override Object newInstance(Object params) {
            if (To instanceof String) {
                String className = (String) To;
                Type toType = getTypeForClassName(className);

                if (toType == null) {
                    throw new BindingException('Apex binding ' + DeveloperName + ' implementation ' + To + ' does not exist');
                }

                Object toObject = toType.newInstance();

                IsProvider = toObject instanceof Provider;
                if (IsProvider) {
                    return ((Provider) toObject).newInstance(params);
                }

                if (params != null) {
                    throw new BindingException('Apex binding ' + DeveloperName + ' implementation ' + className + ' does not implement the Provider interface.');
                }

                return toObject;
            }

            return To;
        }

        private Type getTypeForClassName(String className) {
            return NameSpacePrefix == null ?
                Type.forName(className) :
                Type.forName(NamespacePrefix, className);
        }
    }

    private class VisualForceComponentBinding extends di_Binding {
        public override Object newInstance(Object params) {
            if (To instanceof String) {
                String className = (String) To;
                Type toType = getTypeForClassName(className);

                if (toType == null) {
                    throw new BindingException('Visualforce Component binding ' + DeveloperName + ' implementation ' + className + ' does not exist.');
                }

                Object toObject = toType.newInstance();
                IsProvider = toObject instanceof Provider;

                if (IsProvider) {
                    return ((Provider) toObject).newInstance(params);
                }

                throw new BindingException('Visualforce Component binding ' + DeveloperName + ' must point to a class implementing the Provider interface.');
            }

            return To;
        }

        private Type getTypeForClassName(String className) {
            return NameSpacePrefix == null ?
                Type.forName(className) :
                Type.forName(NamespacePrefix, className);
        }
    }

    private class LightningComponentBinding extends di_Binding {
        public override Object newInstance(Object params) {
            return To;
        }
    }

    private class FlowBinding extends di_Binding {
        public override Object newInstance(Object params) {
            if (To instanceof String) {
                String flowName = (String) To;
                Map<String, Object> flowParams = getFlowParameters(params);
                return new di_Flow(Flow.Interview.createInterview(flowName, flowParams));
            }

            return To;
        }

        private Map<String, Object> getFlowParameters(Object params) {
            if (params instanceof Map<String, Object>) {
                return (Map<String, Object>) params;
            }
            return new Map<String, Object>();
        }
    }
}