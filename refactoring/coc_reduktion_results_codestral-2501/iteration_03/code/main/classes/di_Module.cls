public virtual class di_Module {

    private List<di_Binding> bindings = new List<di_Binding>();
    private di_Binding.BindingType bindingType;
    private String developerName;
    private SObjectType bindingObject;
    private Integer bindingSequence;
    private Object to;
    private Object bindingData;

    public di_Module() {
        init();
    }

    public virtual void configure() { }

    public di_Module type(String value) {
        List<di_Binding.BindingType> bindEnumValues = di_Binding.BindingType.values();

        for(di_Binding.BindingType bindEnumValue : bindEnumValues) {
            if(bindEnumValue.name().equals(value)) {
                bindingType = bindEnumValue;
                return this;
            }
        }

        throw new ModuleException('Binding type ' + value + ' is not valid.');
    }

    public di_Module type(di_Binding.BindingType value) {
        bindingType = value;
        return this;
    }

    public di_Module apex() {
        bindingType = di_Binding.BindingType.Apex;
        return this;
    }

    public di_Module lightningComponent() {
        bindingType = di_Binding.BindingType.LightningComponent;
        return this;
    }

    public di_Module visualforceComponent() {
        bindingType = di_Binding.BindingType.VisualforceComponent;
        return this;
    }

    public di_Module flow() {
        bindingType = di_Binding.BindingType.Flow;
        return this;
    }

    public di_Module module() {
        bindingType = di_Binding.BindingType.Module;
        return this;
    }

    public di_Module bind(Type bindingType) {
        developerName = bindingType.getName();
        return this;
    }

    public di_Module bind(SObjectType sObjectType) {
        bindingObject = sObjectType;
        return this;
    }

    public di_Module bind(String bindingName) {
        developerName = bindingName;
        return this;
    }

    public di_Module sequence(Integer sequence) {
        bindingSequence = sequence;
        return this;
    }

    public di_Module data(Object data) {
        bindingData = data;
        return this;
    }

    public di_Module to(String bindingTo) {
        return toObject(bindingTo);
    }

    public di_Module to(Type bindingTo) {
        return toObject(bindingTo.getName());
    }

    public di_Module toObject(Object to) {
        this.to = to;

        if(to instanceof di_Flow) {
            return flow().addBinding();
        }

        return addBinding();
    }

    public List<di_Binding> getBindings() {
        return this.bindings;
    }

    private di_Module addBinding() {
        di_Binding newBinding = createNewBinding();
        bindings.add(newBinding);
        di_PlatformCache.getInstance().addBindingToPlatformCache(newBinding);
        init();

        return this;
    }

    private di_Binding createNewBinding() {
        return di_Binding.newInstance(
            bindingType,
            developerName,
            bindingObject,
            bindingSequence,
            to,
            bindingData
        );
    }

    private void init() {
        bindingType = di_Binding.BindingType.Apex;
        developerName = null;
        bindingObject = null;
        bindingSequence = null;
        to = null;
    }

    public class ModuleException extends Exception {}
}