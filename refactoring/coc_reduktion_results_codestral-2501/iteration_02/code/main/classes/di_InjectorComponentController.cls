public with sharing class di_InjectorComponentController {

    public String BindingNameValue {set; get;}

    public String BindingIdValue {set; get;}
    public Object ParametersValue {set; get;}

    public ApexPages.Component getInject() {
        di_Binding bindingInfo = getBinding(BindingNameValue);
        return createComponentFromBinding(bindingInfo);
    }

    private ApexPages.Component createComponentFromBinding(di_Binding bindingInfo) {
        switch on bindingInfo.BindingType {
            when Flow {
                return (ApexPages.Component) getInjectorFlowProxyInstance((String) bindingInfo.To, ParametersValue);
            }
            when else {
                return (ApexPages.Component) di_Injector.Org.getInstance(BindingNameValue, ParametersValue);
            }
        }
    }

    public static di_Binding getBinding(String bindingName) {
        return di_Injector.Org.Bindings.byName(bindingName).get()[0];
    }

    @testVisible
    private Object getInjectorFlowProxyInstance(String flowName, Object params) {
        return new Component.di_injectorFlowProxy(
            flowName = flowName,
            inputVariables = params
        );
    }
}