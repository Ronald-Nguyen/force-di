public class di_Injector {
    @testVisible
    private static List<di_BindingConfigWrapper> mock_BindingConfigurationWrappersOuter = null;
    
    public static final di_Injector Org =
        new di_Injector(
            new List<di_Module> {
        new CustomMetadataModule()
    });

    
    @TestVisible
    public di_Binding.Resolver Bindings {get; private set;}

    
    public di_Injector(di_Module module) {
        this(new List<di_Module> { module });
    }

    
    public di_Injector(List<di_Module> modules) {
        Bindings = new di_Binding.Resolver(modules);
    }

    
    public Object getInstance(Type developerNameByType) {
        return getInstance(developerNameByType?.getName()?.toLowerCase(), null);
    }

    
    public Object getInstance(Type developerNameByType, Object params) {
        return getInstance(developerNameByType?.getName()?.toLowerCase(), params);
    }

    
    public Object getInstance(String developerName) {
        return getInstance(developerName, null);
    }

    
    public Object getInstance(String developerName, Object params) {
        if ( String.isBlank(developerName) ) {
            throw new InjectorException('Request for Binding cannot take "developerName" parameter of null');
        }
        List<di_Binding> bindingsFound = this.Bindings.byName( developerName.toLowerCase().trim() ).get();
        if ( bindingsFound == null || bindingsFound.isEmpty() ) {
            throw new InjectorException('Binding for "' + developerName + '" not found');
        }
        return bindingsFound[0].getInstance(params);
    }

    
    public Object getInstance(Type developerNameByType, Schema.SObjectType bindingSObjectType) {
        return getInstance(developerNameByType, bindingSObjectType, null);
    }

    
    public Object getInstance(Type developerNameByType, Schema.SObjectType bindingSObjectType, Object params) {
        return getInstance(developerNameByType == null ? null : developerNameByType.getName().toLowerCase(), bindingSObjectType, params);
    }

    
    public Object getInstance(String developerName, Schema.SObjectType bindingSObjectType, Object params) {
        if ( String.isBlank(developerName) ) {
            throw new InjectorException('Request for Binding cannot take "developerName" parameter of null');
        }

        if ( bindingSObjectType == null ) {
            throw new InjectorException('Request for Binding cannot take "bindingSObjectType" parameter of null');
        }

        List<di_Binding> bindingsFound = this.Bindings.bySObject( bindingSObjectType )
                                         .byName( developerName.toLowerCase().trim() )
                                         .get();

        if ( bindingsFound == null || bindingsFound.isEmpty() || bindingsFound[0] == null) {
            throw new InjectorException('Binding for "' + developerName + '" and SObjectType "' + bindingSObjectType + '" not found');
        }
        return bindingsFound[0].getInstance(params);

    }

    
    @testVisible
    private class CustomMetadataModule extends di_Module {

        private string bindingObjectApiName = null;
        
        
        @testVisible
        private List<di_BindingConfigWrapper> getDIBinding(){
            List<di_BindingConfigWrapper> recordList = new List<di_BindingConfigWrapper>();

            
            if(di_Injector.mock_BindingConfigurationWrappersOuter != null) {
                return di_Injector.mock_BindingConfigurationWrappersOuter;
            }
            
            List<di_Binding__mdt> bindingMDTRec = [SELECT QualifiedAPIName
                                                            , DeveloperName
                                                            , BindingName__c
                                                            , NamespacePrefix
                                                            , Type__c
                                                            , To__c
                                                            , BindingObject__c
                                                            , BindingObject__r.QualifiedApiName
                                                            , BindingObjectAlternate__c
                                                            , BindingSequence__c
                                                         FROM di_Binding__mdt];
            for(di_Binding__mdt records :bindingMDTRec) {
                recordList.add(new di_BindingConfigWrapper(records));
            }
            
            return recordList;
        }

        public override void configure() {
            
            for(di_BindingConfigWrapper bindingConfig :getDIBinding()) {
                bind(bindingConfig.DeveloperName);
                type(bindingConfig.Type);
                configureBindingObject(bindingConfig);
                
                if(bindingConfig.BindingSequence != null) {
                    sequence(Integer.valueOf(bindingConfig.BindingSequence));
                }
                data(bindingConfig);
                to(bindingConfig.To);
            }
        }

        private void configureBindingObject(di_BindingConfigWrapper bindingConfig) {
            if (String.isBlank(bindingConfig.BindingObject) && String.isBlank(bindingConfig.BindingObjectAlternate)) {
                return;
            }

            bindingObjectApiName = String.isNotBlank(bindingConfig.BindingObject)
                                    ? bindingConfig.bindingObjectQualifiedApiName.toLowerCase().trim()
                                    : bindingConfig.BindingObjectAlternate.toLowerCase().trim();
            
            Schema.DescribeSobjectResult[] results = Schema.describeSObjects(new String[] { bindingObjectApiName });
            if(results.size() != 1) {
                throw new InjectorException('Failed to find SObject ' + bindingObjectApiName + ' referenced by binding ' + bindingConfig.DeveloperName);
            }
            bind(results[0].getSObjectType()); 
        }
    }

    public class InjectorException extends Exception {}
}