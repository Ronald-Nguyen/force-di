public class di_Injector {
    @testVisible
    private static List<di_BindingConfigWrapper> mock_BindingConfigurationWrappersOuter = null;
    public static final di_Injector Org = new di_Injector(new List<di_Module> { new CustomMetadataModule() });
    @TestVisible
    public di_Binding.Resolver Bindings {get; private set;}

    public di_Injector(di_Module module) {
        this(new List<di_Module> { module });
    }

    public di_Injector(List<di_Module> modules) {
        Bindings = new di_Binding.Resolver(modules);
    }

    public Object getInstance(String developerName) {
        return getInstance(developerName, null);
    }

    public Object getInstance(String developerName, Object params) {
        if (String.isBlank(developerName)) {
            throw new InjectorException('Request for Binding cannot take "developerName" parameter of null');
        }
        List<di_Binding> bindingsFound = Bindings.byName(developerName.toLowerCase().trim()).get();
        if (bindingsFound.isEmpty()) {
            throw new InjectorException('Binding for "' + developerName + '" not found');
        }
        return bindingsFound[0].getInstance(params);
    }

    public Object getInstance(String developerName, Schema.SObjectType bindingSObjectType, Object params) {
        if (String.isBlank(developerName)) {
            throw new InjectorException('Request for Binding cannot take "developerName" parameter of null');
        }
        if (bindingSObjectType == null) {
            throw new InjectorException('Request for Binding cannot take "bindingSObjectType" parameter of null');
        }
        List<di_Binding> bindingsFound = Bindings.bySObject(bindingSObjectType).byName(developerName.toLowerCase().trim()).get();
        if (bindingsFound.isEmpty() || bindingsFound[0] == null) {
            throw new InjectorException('Binding for "' + developerName + '" and SObjectType "' + bindingSObjectType + '" not found');
        }
        return bindingsFound[0].getInstance(params);
    }

    @testVisible
    private class CustomMetadataModule extends di_Module {
        private String bindingObjectApiName = null;
        @testVisible
        private List<di_BindingConfigWrapper> getDIBinding() {
            return mock_BindingConfigurationWrappersOuter != null ? mock_BindingConfigurationWrappersOuter : [SELECT QualifiedAPIName, DeveloperName, BindingName__c, NamespacePrefix, Type__c, To__c, BindingObject__c, BindingObject__r.QualifiedApiName, BindingObjectAlternate__c, BindingSequence__c FROM di_Binding__mdt].convert(new di_BindingConfigWrapper());
        }

        public override void configure() {
            for (di_BindingConfigWrapper bindingConfig : getDIBinding()) {
                bind(bindingConfig.DeveloperName);
                type(bindingConfig.Type);
                if (String.isNotBlank(bindingConfig.BindingObject) || String.isNotBlank(bindingConfig.BindingObjectAlternate)) {
                    bindingObjectApiName = String.isNotBlank(bindingConfig.BindingObject) ? bindingConfig.bindingObjectQualifiedApiName.toLowerCase().trim() : bindingConfig.BindingObjectAlternate.toLowerCase().trim();
                    Schema.SObjectType sObjectType = Schema.getGlobalDescribe().get(bindingObjectApiName);
                    if (sObjectType == null) {
                        throw new InjectorException('Failed to find SObject ' + bindingObjectApiName + ' referenced by binding ' + bindingConfig.DeveloperName);
                    }
                    bind(sObjectType);
                }
                if (bindingConfig.BindingSequence != null) {
                    sequence(Integer.valueOf(bindingConfig.BindingSequence));
                }
                data(bindingConfig);
                to(bindingConfig.To);
            }
        }
    }

    public class InjectorException extends Exception {}
}